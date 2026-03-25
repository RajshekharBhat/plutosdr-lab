#!/usr/bin/env python3
"""
Experiment 03 — Image over the air (TX)
=========================================
Compress image as JPEG bytes → packetize → QPSK modulate → Pluto TX.

Usage:
  python image_tx.py photo.jpg [--quality 30] [--scheme QPSK]
"""

import sys, io, struct, argparse
import numpy as np
from PIL import Image

sys.path.insert(0, "../../")
from common.pluto_config import connect_tx
from common.modulation   import bits_to_symbols, upsample_and_shape
from common.framing      import build_frame, preamble_symbols

SPS      = 4
PKT_SIZE = 512          # bytes per packet
MARKER   = b"\xDE\xAD"  # 2-byte packet start marker


def image_to_packets(path: str, quality: int) -> list:
    """Compress image and split into byte packets with headers."""
    img = Image.open(path).convert("RGB")
    # Resize to something reasonable for over-the-air
    img.thumbnail((320, 240))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    data = buf.getvalue()
    n_pkts = (len(data) + PKT_SIZE - 1) // PKT_SIZE
    print(f"  Image: {img.size}  JPEG size: {len(data)} bytes  Packets: {n_pkts}")
    packets = []
    for i in range(n_pkts):
        chunk = data[i * PKT_SIZE : (i + 1) * PKT_SIZE]
        # Header: MARKER(2) + total_packets(2) + pkt_index(2) + payload_len(2)
        hdr   = MARKER + struct.pack(">HHH", n_pkts, i, len(chunk))
        packets.append(hdr + chunk)
    return packets


def build_iq(packet: bytes, scheme: str) -> np.ndarray:
    bits   = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
    framed = build_frame(bits, scheme)
    syms   = np.concatenate([preamble_symbols(), bits_to_symbols(framed, scheme)])
    shaped = upsample_and_shape(syms, SPS)
    shaped /= np.max(np.abs(shaped)) * 1.2
    return (shaped * 2**14).astype(np.complex64)


def run(args):
    packets = image_to_packets(args.image, args.quality)
    sdr     = connect_tx(gain_att=args.att)
    print(f"[TX] Sending {len(packets)} packets  scheme={args.scheme}")
    for i, pkt in enumerate(packets):
        iq = build_iq(pkt, args.scheme)
        sdr.tx(iq)
        print(f"\r  Packet {i+1}/{len(packets)}", end="", flush=True)
    sdr.tx_destroy_buffer()
    print("\n[TX] Image transmission complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--quality", type=int,   default=30)
    p.add_argument("--scheme",  default="QPSK", choices=["BPSK","QPSK","QAM16","QAM64"])
    p.add_argument("--att",     type=float, default=-30)
    run(p.parse_args())
