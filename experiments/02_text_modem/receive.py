#!/usr/bin/env python3
"""
Experiment 02 — Classical Text Modem (RX side)
================================================
Full PHY receiver:
  PlutoSDR RX → demodulate_buffer (3-stage carrier recovery) →
  frame detection → CRC check → decode text

Run in a separate terminal while transmit.py is running:
  python receive.py [--scheme QPSK]
"""

import sys
import numpy as np

sys.path.insert(0, "../../")
from common.pluto_config import connect_rx
from common.receiver     import demodulate_buffer
from common.dsp          import estimate_snr_m2m4

SPS = 4


def bits_to_text(bits: np.ndarray) -> str:
    try:
        return np.packbits(bits).tobytes().decode("utf-8", errors="replace")
    except Exception:
        return "<decode error>"


def run():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", default="QPSK",
                   choices=["BPSK", "QPSK", "QAM16", "QAM64"])
    p.add_argument("--gain",   type=float, default=50)
    p.add_argument("--fs",     type=float, default=2e6)
    p.add_argument("--fc",     type=float, default=915e6)
    args = p.parse_args()

    sdr = connect_rx(fc=args.fc, fs=args.fs, gain_db=args.gain,
                     buf_size=2**17)
    print(f"[RX] Listening...  scheme={args.scheme}  "
          f"fc={args.fc/1e6:.3f} MHz")
    print("     Press Ctrl-C to stop.\n")

    n_received = 0
    try:
        while True:
            raw    = sdr.rx()
            frames = demodulate_buffer(raw, args.fs, args.scheme)
            for payload_bits, scheme_r in frames:
                text    = bits_to_text(payload_bits)
                snr_db  = estimate_snr_m2m4(
                    np.frombuffer(payload_bits, dtype=np.uint8).astype(float))
                n_received += 1
                print(f"[RX #{n_received}]  scheme={scheme_r}")
                print(f"  Message : {text}\n")
    except KeyboardInterrupt:
        print(f"\n[RX] Stopped. {n_received} message(s) decoded.")


if __name__ == "__main__":
    run()
