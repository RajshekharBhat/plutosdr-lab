#!/usr/bin/env python3
"""
Experiment 10 — Video streaming over the air (RX)
===================================================
Pluto RX → demodulate → reassemble JPEG frames → save as video / JPEGs.

Usage:
  python video_rx.py [--scheme BPSK] [--out received_video.mp4]

Start this BEFORE video_tx.py.  Press Ctrl-C to stop early; partial video
will still be saved.
"""

import sys, io, struct, time, argparse
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

sys.path.insert(0, "../../")
from common.pluto_config import connect_rx
from common.receiver     import demodulate_buffer

MARKER   = b"\xC0\xFE"
WIDTH, HEIGHT = 320, 240
FPS_OUT  = 10   # output video frame rate for cv2.VideoWriter


def decode_video_packet(raw_bytes: bytes):
    """
    Parse one video packet.  Returns (frame_id, n_pkts, pkt_idx, payload)
    or None on malformed input.
    Header layout: MARKER(2) + frame_id(2) + n_pkts(2) + pkt_idx(2) + plen(2)
    """
    if len(raw_bytes) < 10:
        return None
    if raw_bytes[:2] != MARKER:
        return None
    frame_id, n_pkts, pkt_idx, plen = struct.unpack(">HHHH", raw_bytes[2:10])
    payload = raw_bytes[10 : 10 + plen]
    if len(payload) < plen:
        return None
    return frame_id, n_pkts, pkt_idx, payload


class FrameAssembler:
    """Collect packets and emit complete JPEG bytes when a frame is done."""

    def __init__(self):
        self._pkts   = {}   # frame_id → {pkt_idx: payload}
        self._counts = {}   # frame_id → expected n_pkts

    def add(self, frame_id, n_pkts, pkt_idx, payload) -> bytes | None:
        if frame_id not in self._pkts:
            self._pkts[frame_id]   = {}
            self._counts[frame_id] = n_pkts
        self._pkts[frame_id][pkt_idx] = payload
        if len(self._pkts[frame_id]) == self._counts[frame_id]:
            data = b"".join(self._pkts[frame_id][i]
                            for i in range(self._counts[frame_id]))
            del self._pkts[frame_id]
            del self._counts[frame_id]
            return data
        return None

    @property
    def pending_frames(self):
        return len(self._pkts)


def jpeg_to_bgr(jpeg_bytes: bytes):
    """Decode JPEG bytes to OpenCV BGR array. Returns None on failure."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def run(args):
    if not HAS_CV2:
        sys.exit("[ERROR] opencv-python not installed: pip install opencv-python-headless")

    sdr       = connect_rx(fs=args.fs, buf_size=2**18, gain_db=50)
    assembler = FrameAssembler()

    # cv2.VideoWriter for output
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(args.out, fourcc, FPS_OUT, (WIDTH, HEIGHT))

    print(f"[RX] Listening  scheme={args.scheme}  output={args.out}")
    print(f"     Press Ctrl-C to stop.")

    frames_saved = 0
    pkts_total   = 0
    buffers_rx   = 0
    t0           = time.time()

    try:
        while True:
            raw      = sdr.rx()
            buffers_rx += 1
            decoded  = demodulate_buffer(raw, args.fs, args.scheme)

            for payload_bits, _ in decoded:
                raw_bytes = np.packbits(payload_bits).tobytes()
                pkt       = decode_video_packet(raw_bytes)
                if pkt is None:
                    continue
                frame_id, n_pkts, pkt_idx, payload = pkt
                pkts_total += 1
                jpeg_bytes  = assembler.add(frame_id, n_pkts, pkt_idx, payload)
                print(f"\r  buf={buffers_rx:5d}  pkts={pkts_total:5d}  "
                      f"frames={frames_saved:3d}  pending={assembler.pending_frames}",
                      end="", flush=True)

                if jpeg_bytes is not None:
                    bgr = jpeg_to_bgr(jpeg_bytes)
                    if bgr is not None:
                        bgr = cv2.resize(bgr, (WIDTH, HEIGHT))
                        writer.write(bgr)
                        frames_saved += 1
                        elapsed = time.time() - t0
                        fps     = frames_saved / elapsed
                        print(f"\n  [FRAME {frames_saved:4d}] "
                              f"{len(jpeg_bytes):5d} B  {fps:.2f} fps  "
                              f"{elapsed:.1f} s elapsed")

    except KeyboardInterrupt:
        pass

    writer.release()
    elapsed = time.time() - t0
    print(f"\n[RX] Stopped. {frames_saved} frames saved to {args.out}  "
          f"({pkts_total} packets, {buffers_rx} RX buffers, {elapsed:.1f} s)")
    if frames_saved == 0:
        print("  Tip: check that video_tx.py is running and both Plutos are connected.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", default="BPSK", choices=["BPSK", "QPSK"])
    p.add_argument("--out",    default="received_video.mp4")
    p.add_argument("--fs",     type=float, default=2e6)
    run(p.parse_args())
