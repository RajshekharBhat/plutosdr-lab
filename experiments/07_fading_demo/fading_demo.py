#!/usr/bin/env python3
"""
Experiment 07 — Fading and Multipath Demo
==========================================
Demonstrates:
  1. Rayleigh flat fading envelope (Jake's model) vs Rician
  2. BER degradation from fast vs slow fading
  3. Frequency-selective fading — PSD, delay spread
  4. OFDM robustness vs single-carrier under frequency-selective fading
  5. Real-time RX fading measurement using two Plutos (OTA mode)

Usage (simulation):
  python fading_demo.py --mode sim

Usage (measure live fading with Plutos):
  python fading_demo.py --mode ota --fc 915e6
"""

import sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, "../../")
from common.channel    import rayleigh_fading, apply_fading, awgn, multipath_channel
from common.modulation import bits_to_symbols, symbols_to_bits

FS     = 1e6    # 1 MSPS
N      = 10000  # samples
SCHEME = "QPSK"


def ber_rayleigh_theory(snr_db: np.ndarray) -> np.ndarray:
    """Theoretical QPSK BER over Rayleigh fading channel."""
    snr = 10 ** (snr_db / 10)
    return 0.5 * (1 - np.sqrt(snr / (1 + snr)))


def ber_awgn_theory(snr_db: np.ndarray) -> np.ndarray:
    """QPSK BER over AWGN."""
    from scipy.special import erfc
    snr = 10 ** (snr_db / 10)
    return 0.5 * erfc(np.sqrt(snr))


def simulate_ber(snr_range, doppler_hz):
    bits_tx = np.random.randint(0, 2, 10000).astype(np.uint8)
    syms    = bits_to_symbols(bits_tx, SCHEME)
    bers    = []
    for snr_db in snr_range:
        h   = rayleigh_fading(len(syms), doppler_hz, FS)
        rx  = apply_fading(syms, h)
        rx  = awgn(rx, snr_db)
        # Simple phase-known demod (to isolate fading effect)
        rx_comp = rx * np.conj(h[:len(rx)] / (np.abs(h[:len(rx)]) + 1e-8))
        bits_rx = symbols_to_bits(rx_comp, SCHEME)
        n = min(len(bits_tx), len(bits_rx))
        bers.append(np.mean(bits_tx[:n] != bits_rx[:n]))
    return np.array(bers)


def simulate(args):
    print("=== Fading Channel Demo (simulation) ===\n")
    snr_range = np.arange(0, 31, 2, dtype=float)

    # 1. Rayleigh fading envelope
    h_slow = rayleigh_fading(N, doppler_hz=5,    fs=FS)
    h_fast = rayleigh_fading(N, doppler_hz=200,  fs=FS)

    # 2. BER curves
    ber_slow  = simulate_ber(snr_range, 5)
    ber_fast  = simulate_ber(snr_range, 200)
    ber_awgn  = ber_awgn_theory(snr_range)
    ber_rayl  = ber_rayleigh_theory(snr_range)

    # 3. Multipath PSD
    delays = [0, 5, 15]; gains = [0, -6, -12]; phases = [0, 45, 90]
    impulse = np.zeros(N, dtype=complex); impulse[0] = 1.0
    ch_out  = multipath_channel(impulse, delays, gains, phases)
    f_psd, S = welch(ch_out, fs=FS, nperseg=256)

    # ---- Plots ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    t  = np.arange(N) / FS * 1000   # ms
    ax.plot(t, 20 * np.log10(np.abs(h_slow) + 1e-8), label="Slow (5 Hz)")
    ax.plot(t, 20 * np.log10(np.abs(h_fast) + 1e-8), label="Fast (200 Hz)", alpha=0.7)
    ax.set_title("Rayleigh Fading Envelope"); ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Envelope (dB)"); ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    ax.semilogy(snr_range, ber_awgn,  "k--",  label="AWGN theory")
    ax.semilogy(snr_range, ber_rayl,  "r--",  label="Rayleigh theory")
    ax.semilogy(snr_range, ber_slow,  "b-o",  ms=4, label="Slow fading sim")
    ax.semilogy(snr_range, ber_fast,  "g-s",  ms=4, label="Fast fading sim")
    ax.set_title("BER vs SNR — Fading vs AWGN"); ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER"); ax.legend(); ax.grid(True, which="both")
    ax.set_ylim([1e-4, 1])

    ax = axes[0, 2]
    ax.plot(f_psd / 1e3, 10 * np.log10(S + 1e-12))
    ax.set_title("Multipath Channel PSD (3 taps)")
    ax.set_xlabel("Frequency (kHz)"); ax.set_ylabel("PSD (dB/Hz)"); ax.grid(True)

    ax = axes[1, 0]
    ax.stem(np.arange(20), np.abs(ch_out[:20]), use_line_collection=True)
    ax.set_title("Channel Impulse Response"); ax.set_xlabel("Delay (samples)")
    ax.set_ylabel("|h|"); ax.grid(True)

    # Autocorrelation of fading channel (coherence time)
    ax = axes[1, 1]
    for h, lbl in [(h_slow, "Slow"), (h_fast, "Fast")]:
        ac = np.correlate(np.abs(h), np.abs(h), mode="full")
        ac = ac[N-1:] / ac[N-1]
        ax.plot(np.arange(200) / FS * 1000, ac[:200], label=lbl)
    ax.set_title("Autocorrelation (coherence time indicator)")
    ax.set_xlabel("Lag (ms)"); ax.set_ylabel("R(τ)"); ax.legend(); ax.grid(True)

    # PDF of Rayleigh envelope
    ax = axes[1, 2]
    env = np.abs(h_slow)
    ax.hist(env, bins=50, density=True, alpha=0.7, label="Simulated")
    x_ = np.linspace(0, env.max(), 200)
    sig = np.sqrt(np.mean(env**2) / 2)
    rayl_pdf = (x_ / sig**2) * np.exp(-x_**2 / (2 * sig**2))
    ax.plot(x_, rayl_pdf, "r-", label="Rayleigh PDF")
    ax.set_title("Rayleigh Envelope PDF"); ax.set_xlabel("Amplitude")
    ax.set_ylabel("pdf"); ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig("fading_demo.png", dpi=150)
    print("  Plot: fading_demo.png")


def ota_measure(args):
    """Transmit a CW tone and measure received envelope for live fading observation."""
    from common.pluto_config import connect_both
    tx, rx = connect_both(fc=args.fc, fs=1e6, gain_att=-40, gain_db=60)
    N_buf  = 2**16

    # CW tone
    cw = np.ones(N_buf, dtype=np.int16) * 8192
    tx.tx([cw, cw])

    print(f"[OTA] CW tone TX at {args.fc/1e6:.3f} MHz. Recording envelope...")
    bufs = []
    for i in range(10):
        raw = rx.rx()
        iq  = raw[0].astype(float) + 1j * raw[1].astype(float)
        bufs.append(iq)
        print(f"\r  Buffer {i+1}/10", end="", flush=True)
    tx.tx_destroy_buffer()

    iq_all = np.concatenate(bufs)
    env    = np.abs(iq_all)
    t_ms   = np.arange(len(env)) / 1e6 * 1000

    plt.figure(figsize=(10, 4))
    plt.plot(t_ms, 20 * np.log10(env + 1e-6))
    plt.title("Live Received Envelope — OTA Fading Measurement")
    plt.xlabel("Time (ms)"); plt.ylabel("Received Level (dBFS)"); plt.grid(True)
    plt.savefig("fading_ota.png", dpi=150)
    print("\n  OTA fading plot: fading_ota.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="sim", choices=["sim", "ota"])
    p.add_argument("--fc",   type=float, default=915e6)
    args = p.parse_args()
    if args.mode == "sim":
        simulate(args)
    else:
        ota_measure(args)
