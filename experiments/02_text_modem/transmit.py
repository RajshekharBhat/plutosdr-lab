#!/usr/bin/env python3
"""
Experiment 02 — Classical Text Modem (TX side)
================================================
Full PHY stack:
  CRC framing → BPSK/QPSK/QAM modulation → RRC pulse shaping
  → PlutoSDR TX @ 915 MHz

Usage:
  python transmit.py "Hello, world!"
  python transmit.py --scheme QPSK --file myfile.txt
  python transmit.py --cyclic          # repeat forever (Ctrl-C to stop)
"""

import sys, argparse, time
import numpy as np

sys.path.insert(0, "../../")
from common.pluto_config import connect_tx
from common.modulation   import bits_to_symbols, upsample_and_shape
from common.framing      import build_frame, preamble_symbols

SPS       = 4
GUARD_LEN = 256   # zero-pad guard samples between frames


def text_to_bits(text: str) -> np.ndarray:
    raw = text.encode("utf-8")
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))


def build_tx_buffer(text: str, scheme: str) -> np.ndarray:
    bits    = text_to_bits(text)
    framed  = build_frame(bits, scheme)
    symbols = np.concatenate([preamble_symbols(), bits_to_symbols(framed, scheme)])
    shaped  = upsample_and_shape(symbols, SPS)
    guard   = np.zeros(GUARD_LEN)
    full    = np.concatenate([shaped, guard])
    full   /= np.max(np.abs(full)) * 1.2
    return (full * 2**14).astype(np.int16)


def run(args):
    if args.file:
        with open(args.file, "r") as f:
            text = f.read()
    else:
        text = args.text

    print(f"[TX] Scheme: {args.scheme}")
    print(f"[TX] Message ({len(text)} chars): {text[:80]}{'...' if len(text)>80 else ''}")

    sdr = connect_tx(gain_att=args.att)
    iq  = build_tx_buffer(text, args.scheme)
    n   = 0
    try:
        while True:
            sdr.tx([iq, iq])
            n += 1
            print(f"\r[TX] Frame #{n} sent", end="", flush=True)
            if not args.cyclic:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        sdr.tx_destroy_buffer()
        print(f"\n[TX] Done. {n} frame(s) sent.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("text",         nargs="?", default="Hello from PlutoSDR!")
    p.add_argument("--scheme",     default="QPSK", choices=["BPSK","QPSK","QAM16","QAM64"])
    p.add_argument("--file",       default=None)
    p.add_argument("--att",        type=float, default=-30)
    p.add_argument("--cyclic",     action="store_true")
    run(p.parse_args())
