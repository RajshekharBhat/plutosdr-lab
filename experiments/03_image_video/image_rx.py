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
from common.modulation   import symbols_to_bits, match_filter, rrc_filter
from common.framing      import (preamble_symbols, detect_preamble,
                                 coarse_freq_offset, parse_frame, SFD_BITS)
from common.dsp          import (agc, remove_dc, correct_freq_offset,
                                 gardner_timing_recovery, CostasLoop, FreqLockLoop)

SPS    = 4
MARKER = b"\xDE\xAD"


def decode_packet(raw_bytes: bytes):
    if len(raw_bytes) < 8:
        return None
    if raw_bytes[:2] != MARKER:
        return None
    n_pkts, idx, plen = struct.unpack(">HHH", raw_bytes[2:8])
    payload = raw_bytes[8:8 + plen]
    return n_pkts, idx, payload


def receive_and_decode(sdr, scheme: str, fs: float):
    """Pull one RX buffer and return list of (n_pkts, idx, payload)."""
    raw  = sdr.rx()
    x    = raw.astype(np.complex64)
    x    = remove_dc(x)
    x    = agc(x)
    zc   = preamble_symbols()
    fo   = coarse_freq_offset(x[:len(zc)], fs)
    x    = correct_freq_offset(x, fo, fs)
    fll  = FreqLockLoop(fs)
    x    = fll.step(x)
    mf   = match_filter(x, SPS)
    delay = len(rrc_filter(SPS)) - 1
    syms = gardner_timing_recovery(mf[delay:], SPS)
    costas = CostasLoop(order=4)
    syms = costas.step(syms)
    hits = detect_preamble(syms, threshold=0.5)
    results = []
    for pos, _ in hits:
        start = pos + len(zc)
        bits  = symbols_to_bits(syms[start:start+8192], scheme)
        sfd   = SFD_BITS
        for j in range(min(64, len(bits) - len(sfd))):
            if np.array_equal(bits[j:j+len(sfd)], sfd):
                payload_bits, _, crc_ok = parse_frame(bits[j+len(sfd):])
                if crc_ok and payload_bits is not None:
                    raw_bytes = np.packbits(payload_bits).tobytes()
                    pkt = decode_packet(raw_bytes)
                    if pkt:
                        results.append(pkt)
                break
    return results


def run(args):
    sdr      = connect_rx(fs=2e6, buf_size=2**18, gain_db=50)
    store    = {}
    n_total  = None
    print(f"[RX] Waiting for image packets  scheme={args.scheme}")
    try:
        while True:
            pkts = receive_and_decode(sdr, args.scheme, 2e6)
            for n_pkts, idx, payload in pkts:
                if n_total is None:
                    n_total = n_pkts
                    print(f"  Image has {n_total} packets")
                store[idx] = payload
                pct = len(store) / n_total * 100
                print(f"\r  Received {len(store)}/{n_total} ({pct:.0f}%)", end="", flush=True)
                if len(store) == n_total:
                    print("\n[RX] All packets received! Reconstructing image...")
                    data = b"".join(store[i] for i in range(n_total))
                    img  = Image.open(io.BytesIO(data))
                    img.save(args.out)
                    print(f"  Saved: {args.out}  Size: {img.size}")
                    return
    except KeyboardInterrupt:
        pct = len(store) / (n_total or 1) * 100
        print(f"\n[RX] Interrupted. {len(store)}/{n_total or '?'} packets ({pct:.0f}%)")
        if store:
            data = b"".join(store.get(i, b"") for i in range(n_total or max(store)+1))
            try:
                img = Image.open(io.BytesIO(data))
                img.save(args.out)
                print(f"  Partial image saved: {args.out}")
            except Exception:
                pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", default="QPSK", choices=["BPSK","QPSK","QAM16","QAM64"])
    p.add_argument("--out",    default="received_image.jpg")
    run(p.parse_args())
