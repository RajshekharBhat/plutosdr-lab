#!/usr/bin/env python3
"""
Experiment 05 — OFDMA Multi-User Demo (simulation)
=====================================================
Simulates a downlink OFDMA frame where the host (base station) assigns
different subcarrier groups to 4 users, each with a different message
and modulation scheme.

Demonstrates:
  - Subcarrier assignment (resource block allocation)
  - Per-user modulation
  - Frequency-selective channel effects per user
  - Per-user detection and decoding

Usage:
  python ofdma_demo.py [--snr 20]
"""

import sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "../../")
from common.modulation import bits_to_symbols, symbols_to_bits, bits_per_symbol
from common.channel    import awgn, multipath_channel

N_FFT  = 256
N_CP   = 32

# Resource block allocation: 4 users, each gets 48 contiguous subcarriers
# Guard bands: first 8, last 8, DC (index 128)
GUARD_LOW  = list(range(0, 8))
GUARD_HIGH = list(range(248, 256))
DC         = [128]
INVALID    = set(GUARD_LOW + GUARD_HIGH + DC)

USER_ALLOC = {
    0: (list(range(8,   56)),  "BPSK",  "User A (BPSK)"),
    1: (list(range(56,  104)), "QPSK",  "User B (QPSK)"),
    2: (list(range(104, 152)), "QAM16", "User C (16QAM)"),
    3: (list(range(152, 200)), "QAM64", "User D (64QAM)"),
}

MESSAGES = [
    "Hello from User A!",
    "User B here, QPSK rocks.",
    "User C: 16QAM data burst.",
    "User D: 64QAM high speed!",
]


def text_to_bits(text: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(text.encode("utf-8"), dtype=np.uint8))


def build_ofdma_symbol(messages: list) -> np.ndarray:
    """Build one OFDMA OFDM symbol from per-user messages."""
    fd = np.zeros(N_FFT, dtype=complex)
    user_bits = {}
    for uid, (subcs, scheme, label) in USER_ALLOC.items():
        bps      = bits_per_symbol(scheme)
        n_bits   = len(subcs) * bps
        bits_raw = text_to_bits(messages[uid])
        # Truncate or pad to fit exactly n_bits
        bits     = np.zeros(n_bits, dtype=np.uint8)
        n        = min(len(bits_raw), n_bits)
        bits[:n] = bits_raw[:n]
        syms     = bits_to_symbols(bits, scheme)
        for k, s in zip(subcs, syms):
            fd[k] = s
        user_bits[uid] = bits
    td = np.fft.ifft(fd) * np.sqrt(N_FFT)
    cp = td[-N_CP:]
    return np.concatenate([cp, td]), user_bits, fd


def decode_ofdma_symbol(rx_td: np.ndarray, H_est: np.ndarray = None):
    """Demodulate OFDMA symbol per user."""
    fd = np.fft.fft(rx_td) / np.sqrt(N_FFT)
    if H_est is not None:
        fd = fd / (H_est + 1e-8)
    decoded = {}
    for uid, (subcs, scheme, label) in USER_ALLOC.items():
        syms    = fd[np.array(subcs)]
        bits    = symbols_to_bits(syms, scheme)
        try:
            text = np.packbits(bits).tobytes().decode("utf-8", errors="replace").rstrip("\x00")
        except Exception:
            text = "<error>"
        decoded[uid] = (text, label)
    return decoded, fd


def run(args):
    print(f"=== OFDMA Demo  N_FFT={N_FFT}  N_CP={N_CP}  SNR={args.snr} dB ===\n")

    # Build frame
    td_with_cp, user_bits, fd_tx = build_ofdma_symbol(MESSAGES)

    # Simple flat AWGN channel
    rx = awgn(td_with_cp, args.snr)
    rx_td = rx[N_CP : N_CP + N_FFT]

    decoded, fd_rx = decode_ofdma_symbol(rx_td)

    print("--- Decoded messages ---")
    for uid, (text, label) in decoded.items():
        tx_text = MESSAGES[uid][:len(text)]
        correct = text.strip() == tx_text.strip()
        print(f"  [{label}] {'OK' if correct else 'ERR'}  '{text}'")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # TX spectrum
    ax = axes[0, 0]
    ax.stem(np.abs(fd_tx), use_line_collection=True, markerfmt="C0.", linefmt="C0-")
    for uid, (subcs, scheme, label) in USER_ALLOC.items():
        ax.axvspan(subcs[0], subcs[-1], alpha=0.15,
                   color=f"C{uid+1}", label=label)
    ax.set_title("TX OFDMA Spectrum")
    ax.set_xlabel("Subcarrier"); ax.set_ylabel("|X[k]|")
    ax.legend(fontsize=7)

    # RX spectrum
    ax = axes[0, 1]
    ax.stem(np.abs(fd_rx), use_line_collection=True, markerfmt="C1.", linefmt="C1-")
    ax.set_title(f"RX Spectrum (SNR={args.snr} dB)")
    ax.set_xlabel("Subcarrier"); ax.set_ylabel("|Y[k]|")

    # Constellation per user
    for uid, (subcs, scheme, label) in list(USER_ALLOC.items())[:2]:
        ax = axes[1, uid]
        syms_rx = fd_rx[np.array(subcs)]
        ax.scatter(np.real(syms_rx), np.imag(syms_rx), s=20, alpha=0.8, label="RX")
        ax.set_title(f"{label} Constellation")
        ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("ofdma_demo.png", dpi=150)
    print("\n  Plot: ofdma_demo.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snr", type=float, default=20.0)
    run(p.parse_args())
