#!/usr/bin/env python3
"""
Experiment 02 — Classical Text Modem (RX side)
================================================
Full PHY receiver stack:
  PlutoSDR RX → DC removal → AGC → Coarse CFO → FLL →
  Matched filter → Timing recovery (Gardner) → Phase recovery (Costas) →
  Frame detection → CRC check → Decode text

Run in a separate terminal while transmit.py is running:
  python receive.py [--scheme QPSK]
"""

import sys, time
import numpy as np

sys.path.insert(0, "../../")
from common.pluto_config import connect_rx
from common.modulation   import symbols_to_bits, match_filter, rrc_filter, bits_per_symbol
from common.framing      import (preamble_symbols, detect_preamble,
                                 coarse_freq_offset, parse_frame, SFD_BITS)
from common.dsp          import (agc, remove_dc, correct_freq_offset,
                                 gardner_timing_recovery, CostasLoop,
                                 FreqLockLoop, estimate_snr_m2m4)

SPS = 4


def bits_to_text(bits: np.ndarray) -> str:
    try:
        return np.packbits(bits).tobytes().decode("utf-8", errors="replace")
    except Exception:
        return "<decode error>"


def process_block(raw_iq: np.ndarray, scheme: str, fs: float) -> list:
    """Process one RX buffer. Returns list of decoded strings."""
    x = remove_dc(raw_iq.astype(np.complex64))
    x = agc(x)

    # Coarse CFO from first preamble worth of samples
    zc     = preamble_symbols()
    L_zc   = len(zc)
    fo     = coarse_freq_offset(x[:L_zc], fs)
    x      = correct_freq_offset(x, fo, fs)

    # FLL for residual tracking
    fll    = FreqLockLoop(fs, bw_norm=0.005)
    x      = fll.step(x)

    # Matched filter
    mf     = match_filter(x, SPS)
    delay  = len(rrc_filter(SPS)) - 1
    mf     = mf[delay:]

    # Timing recovery
    syms   = gardner_timing_recovery(mf, SPS, loop_bw=0.005)

    # Preamble detection
    hits   = detect_preamble(syms, threshold=0.5)
    if not hits:
        return []

    # Phase recovery per detected frame
    costas = CostasLoop(order={"BPSK":2,"QPSK":4}.get(scheme, 4), bw_norm=0.003)
    syms   = costas.step(syms)

    snr_db = estimate_snr_m2m4(syms)

    results = []
    for pos, corr in hits:
        start = pos + L_zc
        if start >= len(syms):
            continue
        # extract bits: first 16 = SFD
        sfd_len = len(SFD_BITS) // bits_per_symbol(scheme)
        frame_syms = syms[start : start + 4096]   # generous window
        bits = symbols_to_bits(frame_syms, scheme)

        # Locate SFD
        sfd_bits = SFD_BITS
        for j in range(min(64, len(bits) - len(sfd_bits))):
            if np.array_equal(bits[j:j+len(sfd_bits)], sfd_bits):
                payload_bits, scheme_r, crc_ok = parse_frame(bits[j+len(sfd_bits):])
                if crc_ok and payload_bits is not None:
                    text = bits_to_text(payload_bits)
                    results.append((text, snr_db, corr))
                break
    return results


def run():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scheme",   default="QPSK", choices=["BPSK","QPSK","QAM16","QAM64"])
    p.add_argument("--gain",     type=float, default=50)
    p.add_argument("--fs",       type=float, default=2e6)
    p.add_argument("--fc",       type=float, default=915e6)
    args = p.parse_args()

    sdr = connect_rx(fc=args.fc, fs=args.fs, gain_db=args.gain, buf_size=2**17)
    print(f"[RX] Listening... scheme={args.scheme}  fc={args.fc/1e6:.3f} MHz")
    print("     Press Ctrl-C to stop.\n")

    n_received = 0
    try:
        while True:
            raw = sdr.rx()
            iq  = raw.astype(np.complex64)
            msgs = process_block(iq, args.scheme, args.fs)
            for text, snr, corr in msgs:
                n_received += 1
                print(f"[RX #{n_received}] SNR={snr:.1f} dB  corr={corr:.2f}")
                print(f"  Message: {text}\n")
    except KeyboardInterrupt:
        print(f"\n[RX] Stopped. {n_received} message(s) decoded.")


if __name__ == "__main__":
    run()
