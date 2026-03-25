#!/usr/bin/env python3
"""
Experiment 06 — Channel Estimation Comparison
===============================================
Compares LS, linear-interpolated LS, and MMSE channel estimation
in a frequency-selective channel over OFDM.

Metrics: NMSE of channel estimate, BER vs SNR.

Usage:
  python channel_estimation_demo.py [--snr_max 30]
"""

import sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "../../")
from common.modulation import bits_to_symbols, symbols_to_bits, bits_per_symbol
from common.channel    import awgn, ls_channel_estimate, interpolate_channel, mmse_channel_estimate

N_FFT     = 64
N_CP      = 16
N_PILOTS  = 8
PILOT_IDX = np.linspace(2, N_FFT - 3, N_PILOTS, dtype=int)
DATA_IDX  = np.array([i for i in range(N_FFT) if i not in PILOT_IDX and i != 0])
PILOT_VAL = 1.0 + 0j
SCHEME    = "QPSK"
N_SYMS    = 100   # OFDM symbols per trial


def true_channel_fd(h_td: np.ndarray) -> np.ndarray:
    """Channel frequency response (N_FFT-point DFT of time-domain CIR)."""
    H = np.fft.fft(h_td, n=N_FFT)
    return H


def simulate_ofdm_symbol(h_td: np.ndarray, snr_db: float, scheme: str = SCHEME):
    """Transmit one OFDM symbol through channel, return (rx_td, tx_bits, tx_fd)."""
    bps   = bits_per_symbol(scheme)
    n_bits = len(DATA_IDX) * bps
    bits  = np.random.randint(0, 2, n_bits).astype(np.uint8)
    syms  = bits_to_symbols(bits, scheme)

    fd_tx = np.zeros(N_FFT, dtype=complex)
    fd_tx[PILOT_IDX] = PILOT_VAL
    fd_tx[DATA_IDX]  = syms
    td_tx = np.fft.ifft(fd_tx) * np.sqrt(N_FFT)
    tx_cp = np.concatenate([td_tx[-N_CP:], td_tx])

    # Channel
    rx_cp = np.convolve(tx_cp, h_td)[:len(tx_cp)]
    rx_cp = awgn(rx_cp, snr_db)
    rx_td = rx_cp[N_CP : N_CP + N_FFT]
    return rx_td, bits, fd_tx


def evaluate(snr_db: float, h_td: np.ndarray):
    H_true = true_channel_fd(h_td)
    ber_ls_interp = 0.0
    ber_mmse      = 0.0
    nmse_ls       = 0.0
    nmse_mmse     = 0.0

    # Precompute pilot covariance for MMSE
    R_hh = np.outer(H_true[PILOT_IDX], np.conj(H_true[PILOT_IDX]))
    R_hh = R_hh / np.trace(R_hh)

    for _ in range(N_SYMS):
        rx_td, bits_tx, _ = simulate_ofdm_symbol(h_td, snr_db)
        fd_rx = np.fft.fft(rx_td) / np.sqrt(N_FFT)

        # LS + linear interp
        H_ls_p    = ls_channel_estimate(fd_rx[PILOT_IDX], PILOT_VAL)
        H_ls_all  = interpolate_channel(H_ls_p, PILOT_IDX, N_FFT)
        eq_ls     = fd_rx / (H_ls_all + 1e-8)
        bits_ls   = symbols_to_bits(eq_ls[DATA_IDX], SCHEME)
        ber_ls_interp += np.mean(bits_ls[:len(bits_tx)] != bits_tx[:len(bits_ls)])
        nmse_ls   += np.mean(np.abs(H_ls_all - H_true) ** 2) / np.mean(np.abs(H_true) ** 2)

        # MMSE at pilots → interpolate
        snr_lin   = 10 ** (snr_db / 10)
        H_mmse_p  = mmse_channel_estimate(fd_rx[PILOT_IDX], PILOT_VAL, R_hh, snr_lin)
        H_mmse_all= interpolate_channel(H_mmse_p, PILOT_IDX, N_FFT)
        eq_mmse   = fd_rx / (H_mmse_all + 1e-8)
        bits_mmse = symbols_to_bits(eq_mmse[DATA_IDX], SCHEME)
        ber_mmse  += np.mean(bits_mmse[:len(bits_tx)] != bits_tx[:len(bits_mmse)])
        nmse_mmse += np.mean(np.abs(H_mmse_all - H_true) ** 2) / np.mean(np.abs(H_true) ** 2)

    return (ber_ls_interp / N_SYMS, ber_mmse / N_SYMS,
            nmse_ls / N_SYMS, nmse_mmse / N_SYMS)


def run(args):
    print(f"Channel estimation comparison  ({N_SYMS} OFDM symbols/point)")

    # 4-tap frequency-selective channel
    h_td = np.array([1.0, 0, 0.5 * np.exp(1j * 0.8), 0,
                     0.25 * np.exp(1j * 1.5), 0, 0.1 * np.exp(1j * 0.3)])

    snr_range = np.arange(0, args.snr_max + 1, 2)
    res = [evaluate(s, h_td) for s in snr_range]
    ber_ls, ber_mmse, nmse_ls, nmse_mmse = zip(*res)

    print(f"{'SNR':>6}  {'BER-LS':>10}  {'BER-MMSE':>10}  {'NMSE-LS':>10}  {'NMSE-MMSE':>10}")
    for s, bl, bm, nl, nm in zip(snr_range, ber_ls, ber_mmse, nmse_ls, nmse_mmse):
        print(f"{s:6.0f}  {bl:10.4f}  {bm:10.4f}  {nl:10.4f}  {nm:10.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].semilogy(snr_range, ber_ls,   "o-", label="LS + Interp")
    axes[0].semilogy(snr_range, ber_mmse, "s-", label="MMSE + Interp")
    axes[0].set_title(f"BER vs SNR ({SCHEME}, freq-selective channel)")
    axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("BER")
    axes[0].legend(); axes[0].grid(True, which="both")

    axes[1].semilogy(snr_range, nmse_ls,   "o-", label="LS")
    axes[1].semilogy(snr_range, nmse_mmse, "s-", label="MMSE")
    axes[1].set_title("Normalised MSE of Channel Estimate")
    axes[1].set_xlabel("SNR (dB)"); axes[1].set_ylabel("NMSE")
    axes[1].legend(); axes[1].grid(True, which="both")

    plt.tight_layout()
    plt.savefig("channel_estimation.png", dpi=150)
    print("  Plot: channel_estimation.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snr_max", type=float, default=28)
    run(p.parse_args())
