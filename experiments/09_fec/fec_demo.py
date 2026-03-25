#!/usr/bin/env python3
"""
Experiment 09 — Forward Error Correction: Convolutional Codes
==============================================================
Demonstrates BER improvement from convolutional coding + hard-decision Viterbi
decoding compared to uncoded BPSK transmission over AWGN.

Codes evaluated:
  - Uncoded BPSK                          (theory + simulation baseline)
  - Rate-1/2, K=3, G=[7,5]₈
  - Rate-1/2, K=7, G=[133,171]₈          (NASA standard)
  - Rate-1/3, K=3, G=[7,7,5]₈

Modes:
  --mode sim   BER vs Eb/N0 curves — no hardware needed
  --mode ota   Over-the-air FEC test with PlutoSDR (rate-1/2 K=7, BPSK)

Usage:
  python fec_demo.py --mode sim
  python fec_demo.py --mode sim --ebn0_max 12 --n_bits 50000
  python fec_demo.py --mode ota --fc 915e6 --att -30
"""

import sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc

sys.path.insert(0, "../../")
from common.fec        import ConvCode, CONV_R12_K3, CONV_R12_K7, CONV_R13_K3
from common.modulation import upsample_and_shape, match_filter, rrc_filter
from common.dsp        import agc, remove_dc, correct_freq_offset
from common.framing    import preamble_symbols, coarse_freq_offset, detect_preamble


# ─────────────────────────────────────────────────────────────
#  Code catalogue
# ─────────────────────────────────────────────────────────────

CODES = {
    "r12_k3": (CONV_R12_K3, "Rate 1/2, K=3, G=[7,5]₈"),
    "r12_k7": (CONV_R12_K7, "Rate 1/2, K=7, G=[133,171]₈ (NASA)"),
    "r13_k3": (CONV_R13_K3, "Rate 1/3, K=3, G=[7,7,5]₈"),
}


# ─────────────────────────────────────────────────────────────
#  BER simulation primitives
# ─────────────────────────────────────────────────────────────

def _bpsk_hard(tx_bits: np.ndarray, snr_ch_db: float) -> np.ndarray:
    """BPSK over real AWGN at snr_ch_db.  Returns hard-decided rx bits."""
    snr_lin   = 10.0 ** (snr_ch_db / 10.0)
    symbols   = 2.0 * tx_bits.astype(np.float64) - 1.0   # 0 → −1, 1 → +1
    noise_std = 1.0 / np.sqrt(2.0 * snr_lin)             # σ  (Es = 1)
    rx        = symbols + noise_std * np.random.randn(len(symbols))
    return (rx > 0.0).astype(np.uint8)


def ber_uncoded(ebn0_db: float, n_bits: int) -> float:
    """Simulate uncoded BPSK BER.  For rate-1: SNR_ch = Eb/N0."""
    info = np.random.randint(0, 2, n_bits, dtype=np.uint8)
    rx   = _bpsk_hard(info, ebn0_db)
    return float(np.mean(rx != info))


def ber_theory(ebn0_db: float) -> float:
    """Theoretical uncoded BPSK BER = 0.5 · erfc(√(Eb/N0))."""
    return float(0.5 * erfc(np.sqrt(10.0 ** (ebn0_db / 10.0))))


def ber_coded(ebn0_db: float, code: ConvCode, n_bits: int) -> float:
    """Simulate FEC-coded BPSK BER at given Eb/N0 (info-bit energy).
    Channel SNR per coded bit = code.rate × Eb/N0.
    """
    snr_ch_db = ebn0_db + 10.0 * np.log10(code.rate)
    info      = np.random.randint(0, 2, n_bits, dtype=np.uint8)
    encoded   = code.encode(info)
    rx_hard   = _bpsk_hard(encoded, snr_ch_db)
    decoded   = code.decode(rx_hard, n_bits)
    return float(np.mean(decoded != info))


def _coding_gain_db(ebn0_range, bers_coded, bers_ref, ber_tgt=1e-3):
    """Coding gain (dB) at ber_tgt via log-linear interpolation."""
    def ebn0_at(ebn0s, bers):
        log_b   = np.log10(np.maximum(bers, 1e-12))
        log_tgt = np.log10(ber_tgt)
        idx = np.where(np.diff(np.sign(log_b - log_tgt)))[0]
        if len(idx) == 0:
            return None
        i   = idx[0]
        den = log_b[i + 1] - log_b[i]
        if abs(den) < 1e-12:
            return float(ebn0s[i])
        return float(ebn0s[i] + (log_tgt - log_b[i]) / den *
                     (ebn0s[i + 1] - ebn0s[i]))
    e_ref  = ebn0_at(ebn0_range, bers_ref)
    e_code = ebn0_at(ebn0_range, bers_coded)
    if e_ref is None or e_code is None:
        return float("nan")
    return e_ref - e_code


# ─────────────────────────────────────────────────────────────
#  Simulation mode
# ─────────────────────────────────────────────────────────────

def run_sim(args):
    np.random.seed(args.seed)
    ebn0_range = np.arange(0.0, args.ebn0_max + 0.5, 1.0)

    # Pre-compute theoretical BER
    ref_theory = np.array([ber_theory(e) for e in ebn0_range])

    results = {"uncoded": np.zeros(len(ebn0_range))}
    for k in CODES:
        results[k] = np.zeros(len(ebn0_range))

    hdr = f"{'Eb/N0':>7}  {'Uncoded':>9}  {'R1/2-K3':>9}  {'R1/2-K7':>9}  {'R1/3-K3':>9}"
    print(f"\nSimulation  ({args.n_bits} info bits per point)\n{hdr}")
    print("-" * len(hdr))

    for i, ebn0 in enumerate(ebn0_range):
        results["uncoded"][i] = ber_uncoded(ebn0, args.n_bits)
        row = f"{ebn0:7.1f}  {results['uncoded'][i]:9.5f}"
        for k, (code, _) in CODES.items():
            b = ber_coded(ebn0, code, args.n_bits)
            results[k][i] = b
            row += f"  {b:9.5f}"
        print(row, flush=True)

    # ── Coding gain summary ─────────────────────────────────
    print("\nCoding gain @ BER = 10⁻³ (hard-decision Viterbi vs uncoded BPSK):")
    gains = {}
    for k, (_, label) in CODES.items():
        g = _coding_gain_db(ebn0_range, results[k], ref_theory)
        gains[k] = g
        marker = "~" if np.isnan(g) else f"{g:+.2f} dB"
        print(f"  {label:45s}  {marker}")

    # ── Plot ────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.semilogy(ebn0_range, ref_theory,
                 "k--", lw=1.5, label="Uncoded BPSK (theory)")
    ax1.semilogy(ebn0_range, results["uncoded"],
                 "k.", ms=7, label="Uncoded BPSK (sim)")

    colors  = ["tab:blue", "tab:red", "tab:green"]
    markers = ["o", "s", "^"]
    for (k, (_, label)), col, mk in zip(CODES.items(), colors, markers):
        ax1.semilogy(ebn0_range, results[k], mk + "-",
                     color=col, lw=1.5, ms=5, label=label)

    ax1.axhline(1e-3, color="gray", ls=":", lw=1.0)
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("Bit Error Rate")
    ax1.set_title("BER vs Eb/N0 — Convolutional FEC\n(Hard-Decision Viterbi, BPSK/AWGN)")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_ylim([5e-6, 0.6])
    ax1.set_xlim([0.0, args.ebn0_max])

    # Bar chart — coding gains
    valid_k = [k for k in CODES if not np.isnan(gains.get(k, float("nan")))]
    bar_vals = [gains[k] for k in valid_k]
    bar_lbls = [CODES[k][1].replace("(NASA)", "").strip() for k in valid_k]
    bcols    = [c for c, k in zip(colors, CODES) if k in valid_k]

    bars = ax2.bar(bar_lbls, bar_vals, color=bcols, edgecolor="k", linewidth=0.8)
    ax2.bar_label(bars, fmt="%.1f dB", padding=3, fontsize=10)
    ax2.set_ylabel("Coding Gain (dB) @ BER = 10⁻³")
    ax2.set_title("Coding Gain vs Uncoded BPSK\n(Hard-Decision Viterbi)")
    ax2.grid(True, axis="y", ls="--", alpha=0.5)
    ax2.set_ylim([0.0, max(bar_vals) * 1.4 + 0.5])
    plt.setp(ax2.get_xticklabels(), rotation=12, ha="right", fontsize=8)

    plt.tight_layout()
    out = "fec_ber_curves.png"
    plt.savefig(out, dpi=150)
    print(f"\n  Plot saved: {out}")


# ─────────────────────────────────────────────────────────────
#  OTA mode — PlutoSDR over-the-air FEC test
# ─────────────────────────────────────────────────────────────

SPS        = 4
GUARD_LEN  = 256
N_INFO_OTA = 1024    # information bits per OTA burst


def _build_burst(info_bits: np.ndarray, code: ConvCode | None) -> np.ndarray:
    """
    Build TX IQ burst: [ZC preamble] [BPSK payload] [guard zeros].
    If code is not None, payload = FEC-encoded bits; else raw bits.
    Returns int16 array for sdr.tx().
    """
    payload  = code.encode(info_bits) if code is not None else info_bits
    symbols  = (2.0 * payload.astype(float) - 1.0).astype(complex)
    preamble = preamble_symbols()
    burst    = np.concatenate([preamble, symbols])
    shaped   = upsample_and_shape(burst, SPS)
    guard    = np.zeros(GUARD_LEN)
    full     = np.concatenate([shaped, guard])
    full    /= np.max(np.abs(full)) * 1.2
    return (full * 2**14).astype(np.int16)


def _rx_decode(sdr_rx, n_payload_syms: int, code: ConvCode | None,
               n_info_bits: int, fs: float):
    """Capture one burst and return decoded bits (or None if no preamble)."""
    raw   = sdr_rx.rx()
    raw_c = (raw[0].astype(np.float32)
             + 1j * raw[1].astype(np.float32))
    raw_c = remove_dc(raw_c)
    raw_c = agc(raw_c)

    zc_len = len(preamble_symbols())
    fo_hz  = coarse_freq_offset(raw_c[:zc_len * 2], fs)
    raw_c  = correct_freq_offset(raw_c, fo_hz, fs)

    mf     = match_filter(raw_c, SPS)
    delay  = len(rrc_filter(SPS)) - 1
    syms   = mf[delay::SPS]

    positions = detect_preamble(syms, threshold=0.45)
    if not positions:
        return None

    start   = positions[0][0] + zc_len
    payload = syms[start: start + n_payload_syms]
    if len(payload) < n_payload_syms:
        return None

    hard = (payload.real > 0).astype(np.uint8)
    return code.decode(hard, n_info_bits) if code is not None else hard[:n_info_bits]


def run_ota(args):
    from common.pluto_config import connect_both

    code = CONV_R12_K7
    n_coded = len(code.encode(np.zeros(N_INFO_OTA, dtype=np.uint8)))

    print("=== Experiment 09: FEC OTA Test ===")
    print(f"  Code     : {CODES['r12_k7'][1]}")
    print(f"  Info bits: {N_INFO_OTA}   Coded bits: {n_coded}")
    print(f"  Carrier  : {args.fc/1e6:.1f} MHz    FS: {args.fs/1e6:.1f} MSPS")

    tx, rx = connect_both(fc=args.fc, fs=args.fs,
                          gain_att=args.att, gain_db=args.rx_gain)

    info_bits = np.random.randint(0, 2, N_INFO_OTA, dtype=np.uint8)
    bers = {"uncoded": [], "fec": []}

    for trial in range(1, args.n_trials + 1):
        for label, use_fec, n_syms in [
            ("uncoded", False, N_INFO_OTA),
            ("fec",     True,  n_coded),
        ]:
            iq = _build_burst(info_bits, code if use_fec else None)
            tx.tx([iq, iq])
            bits = _rx_decode(rx, n_syms, code if use_fec else None,
                              N_INFO_OTA, args.fs)
            if bits is None:
                print(f"  [trial {trial:2d}] {label:8s}: preamble not detected")
                continue
            ber = float(np.mean(bits != info_bits))
            bers[label].append(ber)
            print(f"  [trial {trial:2d}] {label:8s}: BER = {ber:.4f}")

    tx.tx_destroy_buffer()

    print("\n──── OTA Summary ────")
    for label in ("uncoded", "fec"):
        if bers[label]:
            print(f"  {label:8s}  mean BER = {np.mean(bers[label]):.4f}"
                  f"  ({len(bers[label])}/{args.n_trials} frames)")
        else:
            print(f"  {label:8s}  no frames decoded")

    if bers["uncoded"] and bers["fec"]:
        ratio = np.mean(bers["uncoded"]) / max(np.mean(bers["fec"]), 1e-6)
        print(f"\n  BER improvement factor (FEC / uncoded): {ratio:.1f}×")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exp 09 — Convolutional FEC BER demo")
    p.add_argument("--mode",     default="sim", choices=["sim", "ota"])
    # sim
    p.add_argument("--ebn0_max", type=float, default=10.0)
    p.add_argument("--n_bits",   type=int,   default=30000)
    p.add_argument("--seed",     type=int,   default=42)
    # ota
    p.add_argument("--fc",       type=float, default=915e6)
    p.add_argument("--fs",       type=float, default=2e6)
    p.add_argument("--att",      type=float, default=-30.0)
    p.add_argument("--rx_gain",  type=float, default=50.0)
    p.add_argument("--n_trials", type=int,   default=5)
    args = p.parse_args()

    if args.mode == "sim":
        run_sim(args)
    else:
        run_ota(args)
