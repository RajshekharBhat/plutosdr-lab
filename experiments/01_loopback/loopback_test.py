#!/usr/bin/env python3
"""
Experiment 01 — Hardware Loopback Test
=======================================
TX (Pluto 1) sends a known BPSK PRBS signal.
RX (Pluto 2) receives and reports:
  - Received power
  - SNR estimate
  - Constellation plot
  - EVM (Error Vector Magnitude)

Run:  python loopback_test.py [--fc 915e6] [--fs 1e6] [--att -30]
"""

import sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "../../")
from common.pluto_config  import connect_both, DEFAULT_FC, DEFAULT_FS
from common.modulation    import bits_to_symbols, symbols_to_bits, upsample_and_shape, match_filter, rrc_filter
from common.dsp           import agc, remove_dc, estimate_snr_m2m4, correct_freq_offset
from common.framing       import preamble_symbols, coarse_freq_offset

SPS   = 4      # samples per symbol
N_SYM = 1024   # number of PRBS symbols


def make_prbs(n: int, seed: int = 0xACE1) -> np.ndarray:
    bits = np.zeros(n, dtype=np.uint8)
    lfsr = seed
    for i in range(n):
        bits[i] = lfsr & 1
        lfsr = (lfsr >> 1) ^ (0xB400 * (lfsr & 1))
    return bits


def run(args):
    print("=== Experiment 01: Hardware Loopback ===")
    tx, rx = connect_both(fc=args.fc, fs=args.fs,
                          gain_att=args.att, gain_db=args.rx_gain)

    # ---- Build TX signal ----
    bits_tx = make_prbs(N_SYM)
    syms_tx = bits_to_symbols(bits_tx, "BPSK")
    preamble = preamble_symbols()
    sig_bb   = np.concatenate([preamble, syms_tx])
    shaped   = upsample_and_shape(sig_bb, SPS)

    # Normalise and scale to int16 range
    shaped  /= np.max(np.abs(shaped)) * 1.2
    iq_tx    = (shaped * 2**14).astype(np.int16)

    # ---- Transmit ----
    tx.tx([iq_tx, iq_tx])   # I and Q same for BPSK (Q=0 effectively via modulator)

    # ---- Receive ----
    raw   = rx.rx()
    raw_c = raw[0].astype(np.float32) + 1j * raw[1].astype(np.float32)
    raw_c = remove_dc(raw_c)
    raw_c = agc(raw_c)

    # ---- Coarse frequency offset ----
    zc_len = len(preamble)
    fo_hz  = coarse_freq_offset(raw_c[:zc_len], args.fs)
    print(f"  Coarse CFO estimate: {fo_hz:.1f} Hz")
    raw_c  = correct_freq_offset(raw_c, fo_hz, args.fs)

    # ---- Matched filter + downsample ----
    mf_out = match_filter(raw_c, SPS)
    delay  = len(rrc_filter(SPS)) - 1
    syms_rx = mf_out[delay + zc_len * SPS :: SPS][:N_SYM]

    # ---- Metrics ----
    snr_db  = estimate_snr_m2m4(syms_rx)
    pwr_dbm = 10 * np.log10(np.mean(np.abs(syms_rx) ** 2) + 1e-12)
    evm_rms = np.sqrt(np.mean(np.abs(syms_rx - syms_tx[:len(syms_rx)]) ** 2)) / \
              np.sqrt(np.mean(np.abs(syms_tx[:len(syms_rx)]) ** 2))
    evm_pct = evm_rms * 100

    print(f"  Received power : {pwr_dbm:.1f} dBFS")
    print(f"  SNR estimate   : {snr_db:.1f} dB")
    print(f"  EVM            : {evm_pct:.1f} %")

    # ---- Constellation plot ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(np.real(syms_rx), np.imag(syms_rx), s=5, alpha=0.4)
    axes[0].set_title("Received Constellation (BPSK)")
    axes[0].set_xlabel("I"); axes[0].set_ylabel("Q")
    axes[0].axhline(0, color="k", lw=0.5); axes[0].axvline(0, color="k", lw=0.5)
    axes[0].set_aspect("equal")

    axes[1].psd(raw_c, NFFT=1024, Fs=args.fs / 1e6)
    axes[1].set_title("RX Power Spectral Density")
    axes[1].set_xlabel("Frequency (MHz)")

    plt.tight_layout()
    plt.savefig("loopback_result.png", dpi=150)
    print("  Plot saved: loopback_result.png")

    tx.tx_destroy_buffer()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fc",       type=float, default=915e6)
    p.add_argument("--fs",       type=float, default=2e6)
    p.add_argument("--att",      type=float, default=-30)
    p.add_argument("--rx-gain",  type=float, default=50)
    run(p.parse_args())
