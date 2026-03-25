#!/usr/bin/env python3
"""
Experiment 10 — Video streaming over the air (TX)
===================================================
Capture frames from webcam or video file → JPEG compress → packetize →
BPSK/QPSK modulate → Pluto TX.

Usage:
  # Live webcam (device 0), 30 frames, BPSK
  python video_tx.py --source 0 --frames 30 --scheme BPSK

  # Video file, QPSK, quality 20
  python video_tx.py --source clip.mp4 --frames 60 --scheme QPSK --quality 20

Two terminals needed: start video_rx.py first, then run this.
"""

import sys, io, struct, time, argparse
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

sys.path.insert(0, "../../")
from common.pluto_config import connect_tx
from common.modulation   import bits_to_symbols, upsample_and_shape
from common.framing      import build_frame, preamble_symbols

SPS      = 4
PKT_SIZE = 256          # bytes per video packet payload
MARKER   = b"\xC0\xFE"  # distinct from image experiment (\xDE\xAD)
WIDTH, HEIGHT = 320, 240


def frame_to_packets(frame_bgr, frame_id: int, quality: int) -> tuple:
    """JPEG-compress one OpenCV frame and split into numbered packets."""
    ok, buf = cv2.imencode(".jpg", frame_bgr,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return [], 0
    data   = buf.tobytes()
    n_pkts = (len(data) + PKT_SIZE - 1) // PKT_SIZE
    packets = []
    for i in range(n_pkts):
        chunk = data[i * PKT_SIZE : (i + 1) * PKT_SIZE]
        # Header: MARKER(2) + frame_id(2) + n_pkts_in_frame(2) +
        #         pkt_idx(2) + payload_len(2)  → 10 bytes total
        hdr = MARKER + struct.pack(">HHHH", frame_id & 0xFFFF,
                                   n_pkts, i, len(chunk))
        packets.append(hdr + chunk)
    return packets, len(data)


def build_iq(packet: bytes, scheme: str) -> np.ndarray:
    bits   = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
    framed = build_frame(bits, scheme)
    syms   = np.concatenate([preamble_symbols(),
                              bits_to_symbols(framed, scheme)])
    shaped = upsample_and_shape(syms, SPS)
    shaped /= np.max(np.abs(shaped)) * 1.2
    return (shaped * 2**14).astype(np.complex64)


def run(args):
    if not HAS_CV2:
        sys.exit("[ERROR] opencv-python not installed: pip install opencv-python-headless")

    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video source: {args.source}")

    sdr = connect_tx(gain_att=args.att)
    print(f"[TX] source={args.source}  scheme={args.scheme}  "
          f"quality={args.quality}  frames={args.frames}")

    frame_id  = 0
    total_pkts = 0
    t0 = time.time()

    while frame_id < args.frames:
        ok, frame = cap.read()
        if not ok:
            if isinstance(source, str):
                break           # video file exhausted
            continue

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        packets, jpg_bytes = frame_to_packets(frame, frame_id, args.quality)
        if not packets:
            frame_id += 1
            continue

        t_frame = time.time()
        for pkt in packets:
            sdr.tx(build_iq(pkt, args.scheme))
        tx_ms  = (time.time() - t_frame) * 1000
        total_pkts += len(packets)
        fps_avg = (frame_id + 1) / (time.time() - t0)
        print(f"  Frame {frame_id:4d}: {jpg_bytes:5d} B → "
              f"{len(packets):2d} pkts  {tx_ms:5.0f} ms  avg {fps_avg:.2f} fps")
        frame_id += 1

    cap.release()
    sdr.tx_destroy_buffer()
    elapsed = time.time() - t0
    print(f"[TX] Done: {frame_id} frames, {total_pkts} packets in {elapsed:.1f} s  "
          f"({frame_id/elapsed:.2f} fps)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source",  default="0",
                   help="Webcam index (0,1,...) or video file path")
    p.add_argument("--frames",  type=int, default=30)
    p.add_argument("--scheme",  default="BPSK",
                   choices=["BPSK", "QPSK"])
    p.add_argument("--quality", type=int, default=20,
                   help="JPEG quality 1-95 (lower = fewer bytes = fewer packets)")
    p.add_argument("--att",     type=float, default=-30,
                   help="TX attenuation in dB (0=max power, -89=min)")
    p.add_argument("--fs",      type=float, default=2e6)
    run(p.parse_args())
