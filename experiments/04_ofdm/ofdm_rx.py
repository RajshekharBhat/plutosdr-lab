#!/usr/bin/env python3
"""
OFDM Over-the-Air Receiver — Experiment 04
============================================
Decodes frames transmitted by ofdm_tx.py.

RX pipeline per buffer:
  1. DC remove + AGC
  2. Timing sync  : normalised cross-correlation with stored ZC preamble
  3. CFO estimate : phase difference between preamble-1 and preamble-2 FFTs
  4. CFO correct  : multiply buffer by exp(-j 2π cfo t)
  5. Channel est  : LS on preamble (ZC known) → linear interp to all subs
  6. Equalise     : ZF per subcarrier for each data symbol
  7. Demodulate   : hard-decision QPSK/QAM, re-assemble bits → text

Usage (open two terminals):
  Terminal 1: python ofdm_tx.py --text "Hello IIT Dharwad!" --scheme QPSK
  Terminal 2: python ofdm_rx.py --scheme QPSK --n_frames 20
"""

import sys, argparse
import numpy as np
from scipy.signal import correlate

sys.path.insert(0, "../../")
from common.modulation   import symbols_to_bits, bits_per_symbol
from common.pluto_config import connect_rx
from common.dsp          import remove_dc, agc

# ─── OFDM parameters (must match ofdm_tx.py) ─────────────────────────────────
N_FFT   = 64
N_CP    = 16
SYM_LEN = N_FFT + N_CP

PILOT_IDX  = np.array([7, 21, 43, 57])
NULL_IDX   = np.concatenate([np.arange(27, 38), [0]])
DATA_IDX   = np.array([i for i in range(N_FFT)
                        if i not in PILOT_IDX and i not in NULL_IDX])
ACTIVE_IDX = np.sort(np.concatenate([DATA_IDX, PILOT_IDX]))


def _zc(length: int, order: int = 1) -> np.ndarray:
    n = np.arange(length)
    return np.exp(-1j * np.pi * order * n * (n + 1) / length)


ZC_ACTIVE = _zc(len(ACTIVE_IDX))
PILOT_VAL = ZC_ACTIVE[np.searchsorted(ACTIVE_IDX, PILOT_IDX)]

# Reference preamble (no CP) for cross-correlation
_fd_ref = np.zeros(N_FFT, dtype=complex)
_fd_ref[ACTIVE_IDX] = ZC_ACTIVE
_td_ref = (np.fft.ifft(_fd_ref) * np.sqrt(N_FFT)).astype(np.complex64)   # (64,)
_REF_POWER = float(np.dot(_td_ref, _td_ref.conj()).real)


# ─── Step 1: Timing ──────────────────────────────────────────────────────────

def find_preamble(rx: np.ndarray, threshold: float = 0.30):
    """
    Paired-peak ZC preamble detector.

    TX sends two identical ZC preamble symbols back-to-back.  We search for
    two cross-correlation peaks separated by exactly SYM_LEN=80 samples.
    The paired metric is: paired[n] = norm_corr[n] + norm_corr[n + SYM_LEN].
    argmax(paired) = start of preamble-1 (no-CP).

    This prevents locking onto preamble-2 at low SNR (both symbols are
    identical, so single-peak detection is ambiguous).

    Returns (pream_nocp_start, mean_peak_corr) or (None, best_val).
    """
    corr = np.abs(correlate(rx, _td_ref, mode='valid'))

    # Amplitude-normalised correlation
    local_e   = np.convolve(np.abs(rx) ** 2, np.ones(N_FFT), mode='valid') + 1e-10
    norm_corr = corr / np.sqrt(local_e * _REF_POWER)

    Nout = len(norm_corr)
    if Nout <= SYM_LEN:
        return None, float(np.max(norm_corr))

    # Paired metric: sum of corr[n] + corr[n + SYM_LEN]
    paired = norm_corr[:Nout - SYM_LEN] + norm_corr[SYM_LEN:]
    best   = int(np.argmax(paired))
    score  = float(paired[best]) / 2     # average of the two peaks

    if score < threshold:
        return None, score
    return best, score


# ─── Step 2: CFO estimation ───────────────────────────────────────────────────

def estimate_cfo(rx: np.ndarray, ps: int, fs: float) -> float:
    """
    Two-preamble CFO estimator.
    ps  : preamble-1 no-CP start index.
    CFO = angle( mean( P2_fd[active] × P1_fd[active]* ) ) × fs / (2π × SYM_LEN)
    Unambiguous range: |CFO| < fs / (2 × SYM_LEN) = ±12.5 kHz @ 2 MSPS.
    """
    p1_td = rx[ps           : ps + N_FFT]
    p2_td = rx[ps + SYM_LEN : ps + SYM_LEN + N_FFT]   # preamble-2, no CP

    if len(p1_td) < N_FFT or len(p2_td) < N_FFT:
        return 0.0

    P1 = np.fft.fft(p1_td) / np.sqrt(N_FFT)
    P2 = np.fft.fft(p2_td) / np.sqrt(N_FFT)

    delta = P2[ACTIVE_IDX] * np.conj(P1[ACTIVE_IDX])
    phase = np.angle(np.mean(delta))
    return float(phase * fs / (2 * np.pi * SYM_LEN))


# ─── Step 3: CFO correction ───────────────────────────────────────────────────

def correct_cfo(rx: np.ndarray, cfo_hz: float, fs: float) -> np.ndarray:
    t = np.arange(len(rx), dtype=np.float32) / fs
    return (rx * np.exp(-1j * 2 * np.pi * cfo_hz * t)).astype(np.complex64)


# ─── Steps 4–6: Channel estimate + equalise + demod ──────────────────────────

def decode_frame(rx: np.ndarray, ps: int, n_data_syms: int, scheme: str):
    """
    Decode one OFDM frame.

    Parameters
    ----------
    rx          : CFO-corrected complex buffer
    ps          : preamble-1 no-CP start index in rx
    n_data_syms : number of data OFDM symbols to decode
    scheme      : 'BPSK' | 'QPSK' | 'QAM16'

    Returns
    -------
    bits   : decoded bit array (concatenated from all data symbols)
    H_all  : channel estimate at all N_FFT subcarriers
    evm_db : rough EVM on pilot subcarriers (dB)
    """
    # ── Channel estimate from preamble-1 ─────────────────────────────────────
    p1_td = rx[ps : ps + N_FFT]
    P1_fd = np.fft.fft(p1_td) / np.sqrt(N_FFT)

    H_active = P1_fd[ACTIVE_IDX] / ZC_ACTIVE          # LS on 52 active subs
    H_all    = (
        np.interp(np.arange(N_FFT), ACTIVE_IDX, np.real(H_active)) +
        1j * np.interp(np.arange(N_FFT), ACTIVE_IDX, np.imag(H_active))
    )

    # EVM on pilot subs (ZC_ACTIVE[pilots_pos] vs H_active[pilots_pos] × ZC_ACTIVE[pilots_pos])
    pilot_pos  = np.searchsorted(ACTIVE_IDX, PILOT_IDX)
    pilot_err  = H_active[pilot_pos] - 1.0    # should be 1 if channel is flat
    evm_db     = float(10 * np.log10(np.mean(np.abs(pilot_err) ** 2) + 1e-12))

    # ── Decode each data symbol ───────────────────────────────────────────────
    all_bits = []
    for i in range(n_data_syms):
        # Data symbol i no-CP start:
        #   ps is the no-CP start of preamble-1, so full-frame start = ps - N_CP.
        #   After 2×SYM_LEN (2 preambles) + i×SYM_LEN + N_CP (skip CP) from frame start:
        #   = (ps - N_CP) + 2×SYM_LEN + i×SYM_LEN + N_CP  = ps + 2×SYM_LEN + i×SYM_LEN
        ds_nocp_start = ps + 2 * SYM_LEN + i * SYM_LEN
        if ds_nocp_start + N_FFT > len(rx):
            break
        ds_td  = rx[ds_nocp_start : ds_nocp_start + N_FFT]
        ds_fd  = np.fft.fft(ds_td) / np.sqrt(N_FFT)
        ds_eq  = ds_fd / (H_all + 1e-8)
        bits   = symbols_to_bits(ds_eq[DATA_IDX], scheme)
        all_bits.append(bits)

    bits_out = np.concatenate(all_bits) if all_bits else np.array([], dtype=np.uint8)
    return bits_out, H_all, evm_db


def _bits_to_text(bits: np.ndarray) -> str:
    n = (len(bits) // 8) * 8
    return np.packbits(bits[:n]).tobytes().decode("utf-8", errors="replace").rstrip("\x00")


# ─── Main receive loop ────────────────────────────────────────────────────────

def run(args):
    scheme   = args.scheme
    bps_sym  = bits_per_symbol(scheme) * len(DATA_IDX)   # bits per data symbol

    print(f"[RX] FC={args.fc/1e6:.3f} MHz  FS={args.fs/1e6:.1f} MSPS  scheme={scheme}  gain={args.gain} dB")
    print(f"[RX] Waiting for OFDM frames (Ctrl-C to stop)...\n")

    sdr   = connect_rx(fc=int(args.fc), fs=int(args.fs),
                       gain_db=args.gain, buf_size=2**17)
    n_ok  = 0
    n_bad = 0

    for frame_num in range(1, args.n_frames + 1):
        raw = sdr.rx().astype(np.complex64)
        raw = remove_dc(raw)
        raw = agc(raw)

        # ── Timing ──────────────────────────────────────────────────────────
        ps, peak = find_preamble(raw, threshold=args.threshold)
        if ps is None:
            n_bad += 1
            print(f"  [{frame_num:3d}]  no preamble  (best corr={peak:.3f})")
            continue

        # ── CFO ──────────────────────────────────────────────────────────────
        cfo_hz = estimate_cfo(raw, ps, args.fs)
        raw_c  = correct_cfo(raw, cfo_hz, args.fs)

        # Re-run timing on CFO-corrected signal (small shift possible)
        ps2, peak2 = find_preamble(raw_c, threshold=args.threshold)
        if ps2 is None:
            ps2, peak2 = ps, peak   # fall back

        # ── How many data symbols fit after the 2 preambles? ─────────────────
        space    = len(raw_c) - ps2 - 2 * SYM_LEN - SYM_LEN   # minus guard
        n_data   = max(1, min(space // SYM_LEN, args.max_syms))

        # ── Decode ───────────────────────────────────────────────────────────
        bits, H_all, evm_db = decode_frame(raw_c, ps2, n_data, scheme)
        text = _bits_to_text(bits)

        n_ok += 1
        print(f"  [{frame_num:3d}]  corr={peak2:.2f}  CFO={cfo_hz:+.0f} Hz  "
              f"syms={n_data}  EVM={evm_db:.1f} dB  \"{text}\"")

    print(f"\n[RX] Done.  Decoded {n_ok}/{frame_num}  |  Missed {n_bad}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OFDM OTA Receiver")
    p.add_argument("--scheme",    default="QPSK", choices=["BPSK", "QPSK", "QAM16"])
    p.add_argument("--fc",        type=float, default=915e6)
    p.add_argument("--fs",        type=float, default=2e6)
    p.add_argument("--gain",      type=float, default=50.0)
    p.add_argument("--n_frames",  type=int,   default=20)
    p.add_argument("--max_syms",  type=int,   default=50)
    p.add_argument("--threshold", type=float, default=0.30,
                   help="Preamble detection normalised-corr threshold (0–1)")
    run(p.parse_args())
