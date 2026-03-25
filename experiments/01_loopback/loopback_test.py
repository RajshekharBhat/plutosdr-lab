#!/usr/bin/env python3
"""
Experiment 01 — Hardware Loopback Test
=======================================
TX (Pluto 1) sends a known BPSK PRBS signal cyclically.
RX (Pluto 2) captures, detects the ZC preamble, demodulates and reports:
  - Coarse CFO estimate (Hz)
  - Received power (dBFS)
  - SNR estimate (M2M4)
  - EVM (RMS %)
  - BPSK BER
  - Constellation plot + PSD plot

Run:  python loopback_test.py [--fc 915e6] [--fs 2e6] [--att -30]
"""

import sys, argparse, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "../../")
from common.pluto_config  import connect_both
from common.modulation    import (bits_to_symbols, symbols_to_bits,
                                  upsample_and_shape, match_filter, rrc_filter)
from common.dsp           import (agc, remove_dc, estimate_snr_m2m4,
                                  correct_freq_offset, CostasLoop)
from common.framing       import preamble_symbols, coarse_freq_offset, detect_preamble

SPS   = 4      # samples per symbol
N_SYM = 1024   # PRBS payload symbols


def make_prbs(n: int, seed: int = 0xACE1) -> np.ndarray:
    bits = np.zeros(n, dtype=np.uint8)
    lfsr = seed
    for i in range(n):
        bits[i] = lfsr & 1
        lfsr = (lfsr >> 1) ^ (0xB400 * (lfsr & 1))
    return bits


def _find_best_preamble(mf: np.ndarray, threshold: float = 0.4):
    """
    Search all SPS timing phases.
    Return (syms, best_hit_pos, best_corr, best_phase) where best = highest single corr.
    """
    best_corr, best_phase, best_pos, best_syms = 0.0, 0, 0, None
    for phase in range(SPS):
        syms = mf[phase::SPS]
        hits = detect_preamble(syms, threshold=threshold)
        for pos, corr in hits:
            if corr > best_corr:
                best_corr, best_phase, best_pos = corr, phase, pos
                best_syms = syms
    return best_syms, best_pos, best_corr, best_phase


def run(args):
    print("=== Experiment 01: Hardware Loopback ===")

    # ---- Connect (cyclic TX so the buffer is always filled) ----
    tx, rx = connect_both(fc=args.fc, fs=args.fs,
                          gain_att=args.att, gain_db=args.rx_gain)
    tx.tx_cyclic_buffer = True

    # ---- Build TX burst: [ZC preamble] [BPSK PRBS] ----
    bits_tx  = make_prbs(N_SYM)
    syms_tx  = bits_to_symbols(bits_tx, "BPSK")
    preamble = preamble_symbols()
    zc_len   = len(preamble)
    sig_bb   = np.concatenate([preamble, syms_tx])
    shaped   = upsample_and_shape(sig_bb, SPS)
    shaped  /= np.max(np.abs(shaped)) * 1.2
    iq_tx    = (shaped * 2**14).astype(np.complex64)

    # ---- Start cyclic TX, wait for AGC to settle ----
    tx.tx(iq_tx)
    time.sleep(0.2)
    raw = rx.rx()
    tx.tx_destroy_buffer()

    # ---- Preprocessing ----
    raw_c = raw.astype(np.complex64)
    raw_c = remove_dc(raw_c)
    raw_c = agc(raw_c)
    delay  = len(rrc_filter(SPS)) - 1

    # ---- Pass 1: find best timing phase + preamble (no CFO correction yet) ----
    mf = match_filter(raw_c, SPS)[delay:]
    syms1, pre_pos1, corr1, phase1 = _find_best_preamble(mf, threshold=0.4)

    if syms1 is not None:
        # Estimate CFO from the located preamble segment
        pre_seg = syms1[pre_pos1 : pre_pos1 + zc_len]
        fo_hz   = coarse_freq_offset(pre_seg, args.fs / SPS)
    else:
        fo_hz = 0.0
        print("  [WARN] Pass-1 preamble not found; CFO set to 0.")

    print(f"  Coarse CFO          : {fo_hz:+.0f} Hz  (pass-1 phase={phase1}, corr={corr1:.3f})")

    # ---- Pass 2: apply CFO correction, find best phase + preamble again ----
    raw_c2 = correct_freq_offset(raw_c, fo_hz, args.fs)
    mf2    = match_filter(raw_c2, SPS)[delay:]
    syms2, pre_pos2, corr2, phase2 = _find_best_preamble(mf2, threshold=0.4)

    if syms2 is None:
        print("  [ERROR] Preamble not found after CFO correction.")
        return

    print(f"  Preamble (pass-2)   : corr={corr2:.3f}  phase={phase2}  sym={pre_pos2}")
    start = pre_pos2 + zc_len
    syms_rx = syms2[start : start + N_SYM]

    if len(syms_rx) < N_SYM // 2:
        print("  [ERROR] Too few symbols after preamble.")
        return

    # ---- Pass 3: fine residual CFO + static phase via BPSK squaring method ----
    # sym^2 removes ±1 modulation:
    #   phase of sq[i] = 2*(theta_static + 2pi*fo_fine*i/fs_sym)
    # Step A — estimate residual frequency offset from phase slope
    n_fine   = min(256, len(syms_rx))
    sq       = syms_rx[:n_fine] ** 2
    phi_diff = np.angle(np.sum(sq[1:] * np.conj(sq[:-1])))   # mean phase step (×2)
    fs_sym   = args.fs / SPS
    fo_fine  = phi_diff * fs_sym / (2 * np.pi * 2)            # /2 because of squaring
    print(f"  Fine residual CFO   : {fo_fine:+.1f} Hz")

    # Apply frequency ramp correction at symbol level
    t_sym   = np.arange(len(syms_rx)) / fs_sym
    syms_rx = syms_rx * np.exp(-1j * 2 * np.pi * fo_fine * t_sym)

    # Step B — estimate and remove static phase offset
    # Squaring gives theta mod 90°: try all 4 candidates, pick the one that
    # minimises imaginary energy (BPSK should sit on the real axis).
    sq_corr    = syms_rx[:n_fine] ** 2
    theta_base = np.angle(np.mean(sq_corr)) / 2    # mod 90°
    best_score  = np.inf
    best_theta  = theta_base
    for k in range(4):
        candidate = theta_base + k * np.pi / 2
        rotated   = syms_rx[:n_fine] * np.exp(-1j * candidate)
        score     = np.mean(np.imag(rotated) ** 2)  # minimum for BPSK on real axis
        if score < best_score:
            best_score = score
            best_theta = candidate
    syms_rx    = syms_rx * np.exp(-1j * best_theta)
    print(f"  Static phase offset : {np.degrees(best_theta):+.1f} deg")

    # ---- Normalise amplitude BEFORE Costas (Costas error ∝ amplitude²) ----
    rms     = np.sqrt(np.mean(np.abs(syms_rx) ** 2))
    syms_rx = syms_rx / (rms + 1e-9)

    # ---- Costas loop for residual phase tracking (unit-amplitude symbols) ----
    costas  = CostasLoop(order=2, bw_norm=0.005)
    syms_rx = costas.step(syms_rx)

    # ---- Metrics ----
    snr_db   = estimate_snr_m2m4(syms_rx)
    pwr_dbfs = 10 * np.log10(rms ** 2 + 1e-12)
    n_cmp    = min(len(syms_rx), len(syms_tx))

    bits_rx  = symbols_to_bits(syms_rx[:n_cmp], "BPSK")
    n_bits   = min(len(bits_rx), n_cmp)
    ber      = np.mean(bits_rx[:n_bits] != bits_tx[:n_bits])
    ber_flip = 1.0 - ber   # BPSK 180° phase ambiguity
    ber_best = min(ber, ber_flip)

    # EVM aligned to correct BPSK polarity (accounts for 180° ambiguity)
    syms_ref = syms_tx[:n_cmp]
    if ber_flip < ber:
        syms_ref = -syms_ref
    evm_rms  = (np.sqrt(np.mean(np.abs(syms_rx[:n_cmp] - syms_ref) ** 2))
                / np.sqrt(np.mean(np.abs(syms_ref) ** 2)))

    print(f"  Received power      : {pwr_dbfs:.1f} dBFS")
    print(f"  SNR estimate (M2M4) : {snr_db:.1f} dB")
    print(f"  EVM (RMS)           : {evm_rms * 100:.1f} %")
    print(f"  BER                 : {ber_best:.4f}  "
          f"({int(ber_best*n_bits)}/{n_bits} errors, "
          f"{'phase-flipped' if ber_flip < ber else 'normal'})")

    # ---- Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].scatter(np.real(syms_rx), np.imag(syms_rx), s=4, alpha=0.4, c="tab:blue")
    axes[0].set_title(f"Received BPSK Constellation\n"
                      f"SNR={snr_db:.1f} dB  EVM={evm_rms*100:.1f}%  BER={ber_best:.4f}")
    axes[0].set_xlabel("I"); axes[0].set_ylabel("Q")
    for v in [-1, 1]:
        axes[0].axvline(v, color="r", lw=0.8, ls="--", alpha=0.6)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    axes[1].psd(raw_c2, NFFT=1024, Fs=args.fs / 1e6)
    axes[1].set_title("RX Power Spectral Density (after CFO correction)")
    axes[1].set_xlabel("Frequency (MHz)")

    plt.tight_layout()
    plt.savefig("loopback_result.png", dpi=150)
    print("  Plot saved          : loopback_result.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fc",       type=float, default=915e6)
    p.add_argument("--fs",       type=float, default=2e6)
    p.add_argument("--att",      type=float, default=-30)
    p.add_argument("--rx-gain",  type=float, default=50)
    run(p.parse_args())
