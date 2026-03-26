#!/usr/bin/env python3
"""
OFDM Over-the-Air Transmitter — Experiment 04
===============================================
Frame structure (all symbols at 1×sample-rate, no SPS shaping):

  [Preamble ZC OFDM × 2] [data OFDM symbols] [zero guard]

  Preamble  : Zadoff-Chu on all 52 active subcarriers → known to RX
  CFO sync  : RX estimates CFO from phase diff between two preamble FFTs
  Ch. est   : RX does LS on preamble (ZC known) → interpolate to all subs
  Pilots    : ZC values also inserted per-symbol on PILOT_IDX (4 subs)

Usage:
  python ofdm_tx.py [--text "Hello!"] [--scheme QPSK] [--fc 915e6]
  python ofdm_tx.py --scheme QAM16 --att -20
"""

import sys, argparse, time
import numpy as np

sys.path.insert(0, "../../")
from common.modulation   import bits_to_symbols, bits_per_symbol
from common.pluto_config import connect_tx

# ─── OFDM parameters (must match ofdm_rx.py) ─────────────────────────────────
N_FFT   = 64
N_CP    = 16
SYM_LEN = N_FFT + N_CP   # 80 samples per symbol

PILOT_IDX = np.array([7, 21, 43, 57])
NULL_IDX  = np.concatenate([np.arange(27, 38), [0]])   # guard + DC
DATA_IDX  = np.array([i for i in range(N_FFT)
                       if i not in PILOT_IDX and i not in NULL_IDX])   # 48 subs
ACTIVE_IDX = np.sort(np.concatenate([DATA_IDX, PILOT_IDX]))            # 52 subs


def _zc(length: int, order: int = 1) -> np.ndarray:
    """Zadoff-Chu sequence of given length."""
    n = np.arange(length)
    return np.exp(-1j * np.pi * order * n * (n + 1) / length)


# Pre-compute ZC pilots: 52-point ZC mapped to active subcarriers
ZC_ACTIVE = _zc(len(ACTIVE_IDX))                                      # (52,) complex
PILOT_VAL = ZC_ACTIVE[np.searchsorted(ACTIVE_IDX, PILOT_IDX)]         # ZC at 4 pilots


def _preamble_td() -> np.ndarray:
    """Build preamble OFDM symbol (with CP): ZC on all 52 active subcarriers."""
    fd = np.zeros(N_FFT, dtype=complex)
    fd[ACTIVE_IDX] = ZC_ACTIVE
    td = np.fft.ifft(fd) * np.sqrt(N_FFT)
    return np.concatenate([td[-N_CP:], td]).astype(np.complex64)   # (80,)


def _data_sym_td(bits: np.ndarray, scheme: str) -> np.ndarray:
    """Pack bits into one OFDM data symbol with ZC pilots."""
    bps = bits_per_symbol(scheme)
    assert len(bits) == len(DATA_IDX) * bps, \
        f"Expected {len(DATA_IDX)*bps} bits, got {len(bits)}"
    syms = bits_to_symbols(bits, scheme)
    fd   = np.zeros(N_FFT, dtype=complex)
    fd[DATA_IDX]  = syms
    fd[PILOT_IDX] = PILOT_VAL
    td = np.fft.ifft(fd) * np.sqrt(N_FFT)
    return np.concatenate([td[-N_CP:], td]).astype(np.complex64)


def build_frame(text: str, scheme: str = "QPSK") -> np.ndarray:
    """
    Build complete OTA OFDM frame.
    Returns complex64 array (not yet scaled to int16).
    """
    bps_sym  = bits_per_symbol(scheme) * len(DATA_IDX)  # bits per data OFDM sym
    raw_bits = np.unpackbits(np.frombuffer(text.encode(), dtype=np.uint8))
    n_syms   = int(np.ceil(len(raw_bits) / bps_sym))
    pad      = n_syms * bps_sym - len(raw_bits)
    bits     = np.concatenate([raw_bits, np.zeros(pad, dtype=np.uint8)])

    pream  = _preamble_td()
    data   = [_data_sym_td(bits[i*bps_sym:(i+1)*bps_sym], scheme) for i in range(n_syms)]
    guard  = np.zeros(SYM_LEN, dtype=np.complex64)

    return np.concatenate([pream, pream] + data + [guard])


def run(args):
    frame   = build_frame(args.text, args.scheme)
    n_data  = (len(frame) // SYM_LEN) - 3   # subtract 2 preambles + 1 guard
    dur_ms  = len(frame) / args.fs * 1e3

    print(f"[TX] Text    : \"{args.text}\"")
    print(f"[TX] Scheme  : {args.scheme}  |  Data syms: {n_data}")
    print(f"[TX] FC={args.fc/1e6:.3f} MHz  FS={args.fs/1e6:.1f} MSPS  att={args.att} dB")
    print(f"[TX] Frame   : {len(frame)} samples ({dur_ms:.2f} ms)  → cycling ...")

    # Scale to Pluto int16 range (keep 6 dB headroom for PAPR)
    peak  = np.max(np.abs(frame)) + 1e-9
    iq    = (frame * (2**13 / peak)).astype(np.complex64)

    sdr = connect_tx(fc=int(args.fc), fs=int(args.fs), gain_att=args.att)
    sdr.tx_cyclic_buffer = True
    sdr.tx(iq)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[TX] Stopping.")
    finally:
        sdr.tx_destroy_buffer()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OFDM OTA Transmitter")
    p.add_argument("--text",   default="Hello IIT Dharwad!  OFDM OTA test 123.")
    p.add_argument("--scheme", default="QPSK", choices=["BPSK", "QPSK", "QAM16"])
    p.add_argument("--fc",     type=float, default=915e6)
    p.add_argument("--fs",     type=float, default=2e6)
    p.add_argument("--att",    type=float, default=-30.0)
    run(p.parse_args())
