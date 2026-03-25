#!/usr/bin/env python3
"""
Experiment 04 — OFDM Modem
============================
Parameters (802.11a-like, scaled down):
  N_FFT  = 64     subcarriers
  N_CP   = 16     cyclic-prefix samples
  N_DATA = 48     data subcarriers
  N_PILOT= 4      pilot subcarriers (indices ±7, ±21)
  N_NULL = 12     DC + guard  (±26 … ±32 + DC)

Supports BPSK/QPSK/QAM16/QAM64 per subcarrier.
Channel estimation: LS with linear interpolation.

Usage (simulation):
  python ofdm_modem.py --mode sim --snr 20 --scheme QPSK

Usage (over the air):
  Terminal 1: python ofdm_modem.py --mode tx --scheme QAM16
  Terminal 2: python ofdm_modem.py --mode rx --scheme QAM16
"""

import sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "../../")
from common.modulation  import bits_to_symbols, symbols_to_bits, bits_per_symbol
from common.channel     import awgn, ls_channel_estimate, interpolate_channel
from common.dsp         import agc, remove_dc

# ---------- OFDM parameters ----------
N_FFT   = 64
N_CP    = 16
PILOT_IDX   = np.array([7, 21, 43, 57])          # pilot subcarrier indices (0-based)
NULL_IDX    = np.concatenate([np.arange(27, 38),  # upper guard
                               [0]])               # DC
DATA_IDX    = np.array([i for i in range(N_FFT)
                         if i not in PILOT_IDX and i not in NULL_IDX])
PILOT_VAL   = 1.0 + 0j    # known BPSK pilot symbol


def bits_to_ofdm_frame(bits: np.ndarray, scheme: str) -> np.ndarray:
    """
    Pack bits into one OFDM symbol.
    Returns time-domain samples (length N_FFT + N_CP).
    bits must be exactly len(DATA_IDX) * bits_per_symbol(scheme) long.
    """
    bps    = bits_per_symbol(scheme)
    assert len(bits) == len(DATA_IDX) * bps
    syms   = bits_to_symbols(bits, scheme)
    fd     = np.zeros(N_FFT, dtype=complex)
    fd[PILOT_IDX] = PILOT_VAL
    fd[DATA_IDX]  = syms
    td     = np.fft.ifft(fd) * np.sqrt(N_FFT)
    cp     = td[-N_CP:]
    return np.concatenate([cp, td])


def ofdm_frame_to_bits(rx_td: np.ndarray, scheme: str,
                        H_est: np.ndarray = None):
    """
    Demodulate one OFDM symbol.
    rx_td  : time-domain (N_FFT + N_CP samples, CP already removed)
    H_est  : channel estimate at all subcarriers (or None → no equalisation)
    Returns (bits, H_data) where H_data is the channel estimate on data subcarriers.
    """
    fd     = np.fft.fft(rx_td) / np.sqrt(N_FFT)
    # Channel estimation from pilots
    H_pilots = ls_channel_estimate(fd[PILOT_IDX], PILOT_VAL)
    H_all    = interpolate_channel(H_pilots, PILOT_IDX, N_FFT)
    # Equalise
    fd_eq  = fd / (H_all + 1e-8)
    data_syms = fd_eq[DATA_IDX]
    bits   = symbols_to_bits(data_syms, scheme)
    return bits, H_all


def build_ofdm_burst(text: str, scheme: str) -> np.ndarray:
    """Build complete OFDM burst from text string."""
    bps     = bits_per_symbol(scheme)
    bits_per_sym = len(DATA_IDX) * bps
    raw_bits = np.unpackbits(np.frombuffer(text.encode(), dtype=np.uint8))
    # Pad to integer number of OFDM symbols
    n_sym   = int(np.ceil(len(raw_bits) / bits_per_sym))
    pad     = n_sym * bits_per_sym - len(raw_bits)
    raw_bits = np.concatenate([raw_bits, np.zeros(pad, dtype=np.uint8)])
    frames  = []
    for i in range(n_sym):
        chunk = raw_bits[i * bits_per_sym : (i + 1) * bits_per_sym]
        frames.append(bits_to_ofdm_frame(chunk, scheme))
    return np.concatenate(frames)


def decode_ofdm_burst(rx: np.ndarray, n_syms: int, scheme: str) -> str:
    bps        = bits_per_symbol(scheme)
    sym_len    = N_FFT + N_CP
    all_bits   = []
    for i in range(n_syms):
        seg    = rx[i * sym_len : i * sym_len + N_FFT + N_CP]
        td     = seg[N_CP:]
        bits, _ = ofdm_frame_to_bits(td, scheme)
        all_bits.append(bits)
    bits_all = np.concatenate(all_bits)
    return np.packbits(bits_all).tobytes().decode("utf-8", errors="replace")


# ---------- Simulation mode ----------

def simulate(args):
    text    = "OFDM over PlutoSDR — channel estimation demo!"
    scheme  = args.scheme
    bps     = bits_per_symbol(scheme)
    bits_per_ofdm = len(DATA_IDX) * bps
    n_syms  = int(np.ceil(len(text.encode()) * 8 / bits_per_ofdm))

    td      = build_ofdm_burst(text, scheme)

    # Multipath channel (3 taps: [1, 0.5e^jpi/4, 0.25e^jpi/2])
    h_mp    = np.array([1.0, 0, 0.5*np.exp(1j*np.pi/4), 0, 0, 0.25*np.exp(1j*np.pi/2)])
    rx      = np.convolve(td, h_mp)[:len(td)]
    rx      = awgn(rx, args.snr)

    decoded = decode_ofdm_burst(rx, n_syms, scheme)
    ber_bits = np.unpackbits(np.frombuffer(text.encode(), dtype=np.uint8))
    dec_bits = np.unpackbits(np.frombuffer(decoded[:len(text)].encode(), errors="replace"), dtype=np.uint8)
    ber = np.mean(ber_bits[:len(dec_bits)] != dec_bits[:len(ber_bits)])

    print(f"OFDM Simulation: scheme={scheme}  SNR={args.snr} dB")
    print(f"  Tx: {text}")
    print(f"  Rx: {decoded.strip()}")
    print(f"  BER: {ber:.4f}")

    # Constellation of one symbol
    sym_len = N_FFT + N_CP
    td_one  = rx[:sym_len][N_CP:]
    fd      = np.fft.fft(td_one) / np.sqrt(N_FFT)
    H_p     = ls_channel_estimate(fd[PILOT_IDX], PILOT_VAL)
    H_a     = interpolate_channel(H_p, PILOT_IDX, N_FFT)
    data_eq = (fd / (H_a + 1e-8))[DATA_IDX]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(np.real(data_eq), np.imag(data_eq), s=8, alpha=0.6)
    axes[0].set_title(f"Equalised Constellation ({scheme})")
    axes[0].set_xlabel("I"); axes[0].set_ylabel("Q"); axes[0].set_aspect("equal")

    axes[1].plot(np.abs(H_a), marker="o", ms=3)
    axes[1].set_title("Channel Magnitude (estimated)")
    axes[1].set_xlabel("Subcarrier"); axes[1].set_ylabel("|H|")

    axes[2].stem(np.abs(h_mp), use_line_collection=True)
    axes[2].set_title("True Channel Impulse Response")
    axes[2].set_xlabel("Tap"); axes[2].set_ylabel("|h|")

    plt.tight_layout()
    plt.savefig("ofdm_sim.png", dpi=150)
    print("  Plot: ofdm_sim.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode",   default="sim", choices=["sim", "tx", "rx"])
    p.add_argument("--scheme", default="QPSK", choices=["BPSK","QPSK","QAM16","QAM64"])
    p.add_argument("--snr",    type=float, default=20.0)
    args = p.parse_args()
    if args.mode == "sim":
        simulate(args)
    else:
        print("OTA TX/RX: see ofdm_tx.py / ofdm_rx.py")
