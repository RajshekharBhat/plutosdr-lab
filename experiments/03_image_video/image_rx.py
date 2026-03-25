#!/usr/bin/env python3
"""
Experiment 03 — Image over the air (RX)
=========================================
Receive packets → reassemble → save JPEG → display info.

Run in a separate terminal while image_tx.py is running:
  python image_rx.py [--scheme QPSK] [--out received.jpg]
"""

import sys, io, struct, argparse
import numpy as np
from PIL import Image

sys.path.insert(0, "../../")
from common.pluto_config import connect_rx
from common.receiver     import demodulate_buffer

MARKER = b"\xDE\xAD"


def decode_packet(raw_bytes: bytes):
    if len(raw_bytes) < 8:
        return None
    if raw_bytes[:2] != MARKER:
        return None
    n_pkts, idx, plen = struct.unpack(">HHH", raw_bytes[2:8])
    payload = raw_bytes[8 : 8 + plen]
    return n_pkts, idx, payload


def run(args):
    sdr     = connect_rx(fs=2e6, buf_size=2**18, gain_db=50)
    store   = {}
    n_total = None
    print(f"[RX] Waiting for image packets  scheme={args.scheme}")
    try:
        while True:
            raw  = sdr.rx()
            decoded = demodulate_buffer(raw, 2e6, args.scheme)
            for payload_bits, _ in decoded:
                raw_bytes = np.packbits(payload_bits).tobytes()
                pkt       = decode_packet(raw_bytes)
                if pkt is None:
                    continue
                n_pkts, idx, payload = pkt
                if n_total is None:
                    n_total = n_pkts
                    print(f"  Image has {n_total} packets")
                store[idx] = payload
                pct = len(store) / n_total * 100
                print(f"\r  Received {len(store)}/{n_total} ({pct:.0f}%)",
                      end="", flush=True)
                if len(store) == n_total:
                    print("\n[RX] All packets received! Reconstructing image...")
                    data = b"".join(store[i] for i in range(n_total))
                    img  = Image.open(io.BytesIO(data))
                    img.save(args.out)
                    print(f"  Saved: {args.out}  Size: {img.size}")
                    return
    except KeyboardInterrupt:
        pct = len(store) / (n_total or 1) * 100
        print(f"\n[RX] Interrupted. {len(store)}/{n_total or '?'} "
              f"packets ({pct:.0f}%)")
        if store:
            data = b"".join(store.get(i, b"") for i in
                            range(n_total or max(store) + 1))
            try:
                img = Image.open(io.BytesIO(data))
                img.save(args.out)
                print(f"  Partial image saved: {args.out}")
            except Exception:
                pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", default="QPSK",
                   choices=["BPSK", "QPSK", "QAM16", "QAM64"])
    p.add_argument("--out", default="received_image.jpg")
    run(p.parse_args())
