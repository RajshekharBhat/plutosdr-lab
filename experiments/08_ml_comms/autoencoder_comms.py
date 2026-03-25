#!/usr/bin/env python3
"""
Experiment 08 — End-to-End Learning (Autoencoder Communications)
=================================================================
Implements the O'Shea & Hoydis (2017) autoencoder approach:
  - Encoder (TX DNN): bits → complex constellation point
  - AWGN channel layer
  - Decoder (RX DNN): received sample → bit estimate

Also includes:
  - Modulation classification (CNN-based AMC)
  - SNR estimation via regression DNN

Requirements: torch (already installed system-wide)

Usage:
  python autoencoder_comms.py --mode train --epochs 50
  python autoencoder_comms.py --mode eval  --snr 10
  python autoencoder_comms.py --mode amc             # modulation classifier
"""

import sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "../../")
from common.modulation import bits_to_symbols, CONSTELLATIONS

# ---------- Autoencoder architecture ----------

class Encoder(nn.Module):
    def __init__(self, M: int, n: int):
        """M = 2^k message size, n = channel uses (complex dims)."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 2 * n),   # 2n = I + Q components
        )
        self.n = n

    def forward(self, x):
        out = self.net(x)                          # (B, 2n)
        # Normalise to unit average power
        out = out / out.norm(dim=1, keepdim=True) * np.sqrt(self.n)
        return out   # (B, 2n) → treat as n complex symbols


class Decoder(nn.Module):
    def __init__(self, M: int, n: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n, 256), nn.ELU(),
            nn.Linear(256, 256),   nn.ELU(),
            nn.Linear(256, M),
        )

    def forward(self, x):
        return self.net(x)   # logits over M classes


class AWGNChannel(nn.Module):
    def __init__(self, snr_db: float):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, x):
        if not self.training:
            return x
        snr    = 10 ** (self.snr_db / 10)
        p_sig  = x.pow(2).mean()
        noise  = torch.randn_like(x) * torch.sqrt(p_sig / (2 * snr))
        return x + noise


class AutoencoderComms(nn.Module):
    def __init__(self, M=16, n=1, snr_db=7.0):
        super().__init__()
        self.M       = M
        self.encoder = Encoder(M, n)
        self.channel = AWGNChannel(snr_db)
        self.decoder = Decoder(M, n)

    def forward(self, msg_onehot):
        tx  = self.encoder(msg_onehot)
        rx  = self.channel(tx)
        out = self.decoder(rx)
        return out, tx


# ---------- Training ----------

def train_autoencoder(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M       = 2 ** args.k   # constellation size
    n       = args.n_uses
    model   = AutoencoderComms(M=M, n=n, snr_db=args.train_snr).to(device)
    optim_  = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    B       = 256   # batch size
    print(f"Training autoencoder: M={M} ({args.k} bits/symbol), n={n}, "
          f"SNR={args.train_snr} dB  device={device}")

    history = []
    for epoch in range(args.epochs):
        model.train()
        msgs = torch.randint(0, M, (B * 50,))
        one_hot = torch.zeros(B * 50, M).scatter_(1, msgs.unsqueeze(1), 1.0)
        ds  = DataLoader(TensorDataset(one_hot, msgs), batch_size=B, shuffle=True)
        ep_loss = 0.0
        for xb, yb in ds:
            xb, yb = xb.to(device), yb.to(device)
            optim_.zero_grad()
            out, _ = model(xb)
            loss   = loss_fn(out, yb)
            loss.backward()
            optim_.step()
            ep_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}  loss={ep_loss/len(ds):.4f}")
        history.append(ep_loss / len(ds))

    torch.save(model.state_dict(), "autoencoder.pt")
    print("  Saved: autoencoder.pt")

    # Evaluate BLER/BER vs SNR
    model.eval()
    snr_range = np.arange(-4, 20, 2, dtype=float)
    bler_ae   = []
    bler_qam  = []

    scheme_name = {4: "QPSK", 8: "QAM16", 16: "QAM64"}.get(M, "QPSK")

    with torch.no_grad():
        for snr_db in snr_range:
            model.channel.snr_db = snr_db
            model.train()   # enable channel noise
            msgs    = torch.randint(0, M, (2000,))
            one_hot = torch.zeros(2000, M).scatter_(1, msgs.unsqueeze(1), 1.0)
            out, _  = model(one_hot)
            preds   = out.argmax(dim=1)
            model.eval()
            bler_ae.append((preds != msgs).float().mean().item())

    # QAM reference (classical hard-decision)
    bits_ref = np.random.randint(0, 2, 2000 * args.k).astype(np.uint8)
    for snr_db in snr_range:
        syms = bits_to_symbols(bits_ref, scheme_name if scheme_name in CONSTELLATIONS else "QPSK")
        snr  = 10 ** (snr_db / 10)
        noise = np.sqrt(1 / (2 * snr)) * (np.random.randn(*syms.shape) +
                                           1j * np.random.randn(*syms.shape))
        rx   = syms + noise
        from common.modulation import symbols_to_bits
        bits_rx = symbols_to_bits(rx, scheme_name if scheme_name in CONSTELLATIONS else "QPSK")
        n = min(len(bits_ref), len(bits_rx))
        bler_qam.append(np.mean(bits_ref[:n] != bits_rx[:n]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].semilogy(snr_range, bler_ae,  "b-o", ms=5, label=f"Autoencoder ({M}-QAM equiv.)")
    axes[0].semilogy(snr_range, bler_qam, "r-s", ms=5, label=f"Classical {scheme_name}")
    axes[0].set_title("BLER/BER vs SNR")
    axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("Error Rate")
    axes[0].legend(); axes[0].grid(True, which="both")

    axes[1].plot(history)
    axes[1].set_title("Training Loss"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CE Loss"); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("autoencoder_result.png", dpi=150)
    print("  Plot: autoencoder_result.png")

    # Show learned constellation
    model.eval()
    msgs     = torch.arange(M)
    one_hot  = torch.zeros(M, M).scatter_(1, msgs.unsqueeze(1), 1.0)
    with torch.no_grad():
        _, tx_syms = model(one_hot)
    tx_np = tx_syms.numpy()
    plt.figure(figsize=(5, 5))
    plt.scatter(tx_np[:, 0], tx_np[:, 1], c=np.arange(M), cmap="tab20", s=100)
    for i in range(M):
        plt.annotate(str(i), (tx_np[i, 0], tx_np[i, 1]), fontsize=8)
    plt.title(f"Learned Constellation (M={M}, n={n})")
    plt.xlabel("I"); plt.ylabel("Q"); plt.axis("equal"); plt.grid(True)
    plt.savefig("learned_constellation.png", dpi=150)
    print("  Constellation: learned_constellation.png")


# ---------- Modulation classifier (AMC) ----------

class AMCNet(nn.Module):
    """Simple 1D CNN for modulation classification."""
    def __init__(self, n_classes: int = 4, seq_len: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 16, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):   # x: (B, 2, L)
        return self.fc(self.conv(x).flatten(1))


def train_amc(args):
    from common.modulation import bits_to_symbols
    schemes   = ["BPSK", "QPSK", "QAM16", "QAM64"]
    L         = 128    # samples per observation
    N_train   = 500    # per class (2000 total) — fast demo; use 2000 for better accuracy
    snr_db    = 15.0

    def make_dataset(n_per_class):
        """Fully vectorised: draw random constellation points directly."""
        from common.modulation import CONSTELLATIONS
        X_all, Y_all = [], []
        snr_lin   = 10 ** (snr_db / 10)
        noise_std = np.sqrt(1 / (2 * snr_lin))
        for label, scheme in enumerate(schemes):
            table = CONSTELLATIONS[scheme][0]       # complex constellation table
            M     = len(table)
            # Draw random symbol indices, look up constellation points
            idx   = np.random.randint(0, M, (n_per_class, L))
            syms  = table[idx]                      # (n, L) complex
            noise = noise_std * (np.random.randn(n_per_class, L) +
                                 1j * np.random.randn(n_per_class, L))
            rx    = syms + noise                    # (n, L)
            iq    = np.stack([np.real(rx), np.imag(rx)],
                              axis=1).astype(np.float32)   # (n, 2, L)
            X_all.append(iq)
            Y_all.extend([label] * n_per_class)
        return (torch.from_numpy(np.concatenate(X_all)),
                torch.tensor(Y_all, dtype=torch.long))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training AMC classifier: {schemes}  SNR={snr_db} dB  "
          f"N={N_train}/class  epochs={args.epochs}  device={device}")
    Xtr, Ytr = make_dataset(N_train)
    Xte, Yte = make_dataset(200)

    model   = AMCNet(n_classes=len(schemes), seq_len=L).to(device)
    opt     = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    ds      = DataLoader(TensorDataset(Xtr, Ytr), batch_size=256, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in ds:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            l = loss_fn(model(xb), yb)
            l.backward()
            opt.step()
            total_loss += l.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
                  f"loss={total_loss/len(ds):.4f}")

    model.eval()
    with torch.no_grad():
        Xte_d = Xte.to(device)
        acc = (model(Xte_d).argmax(1).cpu() == Yte).float().mean().item()
    print(f"  AMC test accuracy: {acc*100:.1f}%")
    model.cpu()
    torch.save(model.state_dict(), "amc_model.pt")
    print("  Saved: amc_model.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       default="train", choices=["train","eval","amc"])
    p.add_argument("--k",          type=int,   default=4,    help="bits per symbol log2(M)")
    p.add_argument("--n_uses",     type=int,   default=1,    help="channel uses")
    p.add_argument("--train_snr",  type=float, default=7.0)
    p.add_argument("--snr",        type=float, default=10.0)
    p.add_argument("--epochs",     type=int,   default=50)
    args = p.parse_args()
    if args.mode == "train":
        train_autoencoder(args)
    elif args.mode == "amc":
        train_amc(args)
    else:
        print("Use --mode train first, then --mode eval")
