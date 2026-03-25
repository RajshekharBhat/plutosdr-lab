#!/usr/bin/env python3
"""
Experiment 07 — Channel Modelling and Fading Demo
===================================================
A complete walk-through of wireless channel characterisation:

  1. Power Delay Profile (PDP) → RMS delay spread σ_τ → coherence bandwidth B_c
  2. Channel frequency response — flat vs frequency-selective
  3. Rayleigh vs Rician fading (K = 0, 1, 5, 10 dB)
  4. Jake's Doppler spectrum → coherence time T_c
  5. Slow vs fast fading envelopes
  6. Fading classification: flat/slow vs flat/fast vs selective/slow vs selective/fast
  7. BER under fading — AWGN theory vs flat Rayleigh theory vs Rayleigh sim vs freq-selective sim
  8. OFDM robustness vs single-carrier under frequency-selective fading

Usage:
  python fading_demo.py --mode sim
  python fading_demo.py --mode ota --fc 915e6

Outputs:
  fading_structure.png   — PDP, freq response, Rayleigh/Rician, classification grid
  fading_doppler.png     — Doppler PSD, coherence time, BER curves, OFDM vs SC
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import welch, fftconvolve
from scipy.special import erfc, i0

sys.path.insert(0, "../../")
from common.channel import (
    rayleigh_fading, rician_fading, apply_fading, awgn,
    multipath_channel, pdp_stats, jakes_doppler_psd
)
from common.modulation import bits_to_symbols, symbols_to_bits

# ── Global constants ──────────────────────────────────────────────────────────
FS     = 1e6        # sample rate 1 MSPS
N      = 20_000     # samples for envelope sims
SCHEME = "BPSK"

# ── Channel definitions ───────────────────────────────────────────────────────
# 3-tap frequency-selective channel (NLOS indoor)
H_SEL_DELAYS = [0, 5, 15]           # samples @ 1 MSPS  → 0, 5, 15 μs
H_SEL_GAINS  = [0.0, -6.0, -15.0]  # dB
H_SEL_PHASES = [0.0, 45.0, 90.0]   # degrees

# Near-flat channel (strong LOS + weak echo 1 sample away)
H_FLAT_DELAYS = [0, 1]
H_FLAT_GAINS  = [0.0, -25.0]
H_FLAT_PHASES = [0.0, 180.0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_cir(delays, gains_db, phases_deg):
    """Build a complex CIR vector from tap specification."""
    h = np.zeros(max(delays) + 1, dtype=complex)
    for d, g, p in zip(delays, gains_db, phases_deg):
        h[d] += 10 ** (g / 20) * np.exp(1j * np.deg2rad(p))
    return h


def _ber_awgn_theory(snr_db):
    """BPSK BER over AWGN: P_b = 0.5 * erfc(sqrt(SNR))."""
    snr = 10 ** (np.asarray(snr_db) / 10)
    return 0.5 * erfc(np.sqrt(snr))


def _ber_rayleigh_theory(snr_db):
    """Theoretical BPSK BER over flat Rayleigh fading with perfect CSI."""
    snr = 10 ** (np.asarray(snr_db) / 10)
    return 0.5 * (1 - np.sqrt(snr / (1 + snr)))


# ── Figure 1: fading_structure.png ───────────────────────────────────────────

def plot_structure(h_sel, h_flat, stats_sel, stats_flat):
    """
    2×3 figure:
      [0,0] PDP  [0,1] Freq response  [0,2] Rayleigh slow/fast envelopes
      [1,0] Rayleigh vs Rician traces  [1,1] Envelope PDFs  [1,2] Classification grid
    """
    mean_d_sel, rms_d_sel, Bc50_sel, Bc90_sel, ntaps_sel = stats_sel

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Channel Modelling: Structure & Fading Distributions", fontsize=13, y=1.01)

    # ── [0,0] Power Delay Profile ────────────────────────────────────────────
    ax = axes[0, 0]
    pdp_lin  = np.abs(h_sel) ** 2
    pdp_db   = 10 * np.log10(pdp_lin + 1e-30)
    delays_us = np.arange(len(h_sel)) / FS * 1e6   # μs

    markerline, stemlines, baseline = ax.stem(delays_us, pdp_db)
    plt.setp(stemlines,   linewidth=1.5)
    plt.setp(markerline,  markersize=6)
    plt.setp(baseline,    linewidth=0.5, color='k')

    ax.axvline(mean_d_sel * 1e6, color='orange', linestyle='--',
               label=f"Mean delay {mean_d_sel*1e6:.2f} μs")
    ax.axvline((mean_d_sel + rms_d_sel) * 1e6, color='red', linestyle=':',
               label=f"Mean + σ_τ ({rms_d_sel*1e6:.2f} μs)")
    ax.set_title("Power Delay Profile (3-tap NLOS indoor)")
    ax.set_xlabel("Delay (μs)")
    ax.set_ylabel("Power (dB)")
    ax.legend(fontsize=8)
    ax.grid(True)
    print(f"  σ_τ = {rms_d_sel*1e6:.2f} μs,  "
          f"B_c(50%) = {Bc50_sel/1e3:.1f} kHz,  "
          f"B_c(90%) = {Bc90_sel/1e3:.1f} kHz")

    # ── [0,1] Channel Frequency Response ─────────────────────────────────────
    ax = axes[0, 1]
    NFFT = 1024
    freqs_kHz = np.fft.fftfreq(NFFT, 1 / FS) / 1e3

    H_sel_f  = np.fft.fft(h_sel,  n=NFFT)
    H_flat_f = np.fft.fft(h_flat, n=NFFT)

    pos_mask = (freqs_kHz >= 0) & (freqs_kHz <= 500)
    ax.plot(freqs_kHz[pos_mask], np.abs(H_sel_f[pos_mask]),
            label="Selective (3-tap)", linewidth=1.5)
    ax.plot(freqs_kHz[pos_mask], np.abs(H_flat_f[pos_mask]),
            label="Near-flat (2-tap)", linewidth=1.5, linestyle='--')
    ax.axhline(Bc50_sel / 1e3 * 0 + 1, color='grey', linewidth=0.8, linestyle=':')
    # Mark coherence BW as vertical line pair
    ax.axvspan(0, Bc50_sel / 1e3, alpha=0.08, color='green',
               label=f"B_c(50%)={Bc50_sel/1e3:.0f} kHz")
    ax.set_title("Channel Freq Response |H(f)|")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("|H(f)|")
    ax.legend(fontsize=8)
    ax.grid(True)

    # ── [0,2] Rayleigh slow/fast envelopes ───────────────────────────────────
    ax = axes[0, 2]
    h_slow = rayleigh_fading(N, 5,   FS)
    h_fast = rayleigh_fading(N, 200, FS)
    NSHOW  = 5000
    t_ms   = np.arange(NSHOW) / FS * 1e3

    ax.plot(t_ms, 20 * np.log10(np.abs(h_slow[:NSHOW]) + 1e-12),
            label="Slow (f_d=5 Hz)", linewidth=1.0)
    ax.plot(t_ms, 20 * np.log10(np.abs(h_fast[:NSHOW]) + 1e-12),
            label="Fast (f_d=200 Hz)", linewidth=0.8, alpha=0.8)
    ax.set_title("Rayleigh Fading Envelope")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Envelope (dB)")
    ax.legend(fontsize=8)
    ax.grid(True)

    # ── [1,0] Rayleigh vs Rician time traces ──────────────────────────────────
    ax = axes[1, 0]
    K_db_vals  = [0, 3, 10]
    K_labels   = ["Rayleigh (K=−∞)", "Rician K=3 dB", "Rician K=10 dB"]

    for K_db, lbl in zip(K_db_vals, K_labels):
        K_lin = 10 ** (K_db / 10) if K_db > 0 else 0.0
        h_k   = rician_fading(N, 10, FS, K_lin)
        ax.plot(t_ms, np.abs(h_k[:NSHOW]), label=lbl, linewidth=0.9)

    ax.set_title("Rayleigh (K=0) vs Rician fading")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("|h(t)|")
    ax.legend(fontsize=8)
    ax.grid(True)

    # ── [1,1] Envelope PDFs ───────────────────────────────────────────────────
    ax = axes[1, 1]
    K_db_pdf = [0, 3, 7, 15]
    colors   = ['C0', 'C1', 'C2', 'C3']
    r_th     = np.linspace(0, 3.5, 500)
    Omega    = 1.0   # unit mean power

    for K_db, col in zip(K_db_pdf, colors):
        K_lin  = 0.0 if K_db <= 0 else 10 ** (K_db / 10)
        h_env  = rician_fading(100_000, 10, FS, K_lin)
        env    = np.abs(h_env)
        lbl    = f"K={K_db} dB" if K_db > 0 else "K=−∞ (Rayleigh)"
        ax.hist(env, bins=80, density=True, alpha=0.35, color=col, label=lbl)

        if K_lin <= 0.0:
            # Rayleigh PDF
            pdf_th = (2 * r_th / Omega) * np.exp(-r_th ** 2 / Omega)
        else:
            # Rician PDF
            pdf_th = (
                (2 * r_th * (K_lin + 1) / Omega)
                * np.exp(-K_lin - (K_lin + 1) * r_th ** 2 / Omega)
                * i0(2 * r_th * np.sqrt(K_lin * (K_lin + 1) / Omega))
            )
        ax.plot(r_th, pdf_th, color=col, linewidth=1.5)

    ax.set_title("Envelope PDF: Rayleigh vs Rician (K-factor)")
    ax.set_xlabel("Envelope amplitude |h|")
    ax.set_ylabel("PDF")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 3.0])
    ax.grid(True)

    # ── [1,2] Fading Classification Grid ─────────────────────────────────────
    ax = axes[1, 2]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Fading Channel Classification", fontsize=10)

    cell_data = [
        # (x0, y0, w, h,  color,          label)
        (0.0, 0.5, 0.5, 0.5, '#d4f4dd', "Flat & Slow\n(ideal AWGN-like)"),
        (0.5, 0.5, 0.5, 0.5, '#fff3cd', "Flat & Fast\n(ICI in OFDM)"),
        (0.0, 0.0, 0.5, 0.5, '#ffe0b2', "Selective & Slow\n(ISI, OFDM robust)"),
        (0.5, 0.0, 0.5, 0.5, '#ffcdd2', "Selective & Fast\n(worst case)"),
    ]
    for (x0, y0, w, h_rect, color, label) in cell_data:
        patch = mpatches.FancyBboxPatch(
            (x0 + 0.01, y0 + 0.01), w - 0.02, h_rect - 0.02,
            boxstyle="round,pad=0.02", facecolor=color, edgecolor='grey', linewidth=1.2
        )
        ax.add_patch(patch)
        ax.text(x0 + w / 2, y0 + h_rect / 2, label,
                ha='center', va='center', fontsize=8.5, fontweight='normal',
                wrap=True)

    # Axis labels
    ax.text(0.25, 1.01, "Flat\n(BW ≪ B_c)", ha='center', va='bottom',
            fontsize=8, color='darkblue', fontweight='bold')
    ax.text(0.75, 1.01, "Freq-Selective\n(BW ≫ B_c)", ha='center', va='bottom',
            fontsize=8, color='darkblue', fontweight='bold')
    ax.text(-0.02, 0.75, "Slow\n(T_s ≪ T_c)", ha='right', va='center',
            fontsize=8, color='darkgreen', fontweight='bold', rotation=90)
    ax.text(-0.02, 0.25, "Fast\n(T_s ≫ T_c)", ha='right', va='center',
            fontsize=8, color='darkgreen', fontweight='bold', rotation=90)

    # Mark our channel: Selective/Slow → bottom-left = [0.0, 0.0] box
    ax.plot(0.25, 0.25, 'r*', markersize=16, zorder=10,
            label="Our channel\n(1 MSPS, f_d=5 Hz)")
    ax.legend(loc='lower right', fontsize=7, framealpha=0.8)

    fig.tight_layout()
    out = "fading_structure.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")
    return h_slow, h_fast


# ── Figure 2: fading_doppler.png ─────────────────────────────────────────────

def plot_doppler(h_slow, h_fast, B_c_sel, T_c_slow, T_c_fast, stats_sel, stats_flat):
    """
    2×3 figure:
      [0,0] Doppler PSD  [0,1] Autocorrelation  [0,2] BER curves
      [1,0] OFDM vs SC   [1,1] Rician K BER     [1,2] Summary text
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Channel Modelling: Doppler, BER, OFDM Robustness", fontsize=13, y=1.01)

    # ── [0,0] Jake's Doppler PSD ──────────────────────────────────────────────
    ax = axes[0, 0]
    # Welch on complex h
    f_slow_w, P_slow_w = welch(h_slow, fs=FS, nperseg=4096)
    f_fast_w, P_fast_w = welch(h_fast, fs=FS, nperseg=4096)

    # Normalise to peak
    P_slow_w = P_slow_w / (np.max(P_slow_w) + 1e-30)
    P_fast_w = P_fast_w / (np.max(P_fast_w) + 1e-30)

    ax.semilogy(f_slow_w, np.abs(P_slow_w) + 1e-10, color='C0', alpha=0.7,
                label="Welch, f_d=5 Hz")
    ax.semilogy(f_fast_w, np.abs(P_fast_w) + 1e-10, color='C1', alpha=0.7,
                label="Welch, f_d=200 Hz")

    # Theoretical Jake's PSD
    f_th = np.linspace(-300, 300, 3000)
    for f_d, col, ls in [(5, 'C0', '--'), (200, 'C1', '--')]:
        jk = jakes_doppler_psd(f_th, f_d)
        # Normalise
        mx = np.max(jk[np.isfinite(jk)]) if np.any(np.isfinite(jk)) else 1
        jk_n = np.where(np.isfinite(jk), jk / mx, 0)
        ax.plot(f_th, jk_n + 1e-10, color=col, linestyle=ls, linewidth=1.5,
                label=f"Jake's theory f_d={f_d} Hz")

    ax.axvline( 5,   color='C0', linestyle=':', linewidth=1)
    ax.axvline( 200, color='C1', linestyle=':', linewidth=1)
    ax.set_xlim([-250, 250])
    ax.set_ylim([1e-5, 2])
    ax.set_title("Doppler PSD (Welch estimate vs Jake's theory)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalised PSD")
    ax.legend(fontsize=7)
    ax.grid(True, which='both')

    # ── [0,1] Autocorrelation and Coherence Time ──────────────────────────────
    ax = axes[0, 1]
    LAG_MAX = 500   # samples → 0.5 ms at 1 MSPS
    lag_ms  = np.arange(LAG_MAX) / FS * 1e3

    for h_ch, lbl, col, f_d_val in [
        (h_slow, f"Slow (f_d=5 Hz), T_c={T_c_slow*1e3:.1f} ms",  'C0', 5),
        (h_fast, f"Fast (f_d=200 Hz), T_c={T_c_fast*1e3:.2f} ms", 'C1', 200),
    ]:
        ac = np.correlate(np.abs(h_ch), np.abs(h_ch), mode='full')
        ac = ac[len(h_ch) - 1:]
        ac = ac / (ac[0] + 1e-30)
        ax.plot(lag_ms, ac[:LAG_MAX], label=lbl, linewidth=1.5)

    T_c_slow_ms = T_c_slow * 1e3
    T_c_fast_ms = T_c_fast * 1e3
    ax.axvline(T_c_slow_ms, color='C0', linestyle='--', linewidth=1.0,
               label=f"T_c slow = {T_c_slow_ms:.1f} ms")
    ax.axvline(T_c_fast_ms, color='C1', linestyle='--', linewidth=1.0,
               label=f"T_c fast = {T_c_fast_ms:.2f} ms")
    ax.axhline(0.423, color='grey', linestyle=':', linewidth=0.8, label="0.423 threshold")
    ax.set_title("Envelope Autocorrelation (Coherence Time)")
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("R(τ) / R(0)")
    ax.legend(fontsize=7)
    ax.grid(True)
    ax.set_xlim([0, lag_ms[-1]])

    # ── [0,2] BER vs SNR: AWGN vs Rayleigh ────────────────────────────────────
    ax = axes[0, 2]
    snr_range = np.arange(0, 31, 2, dtype=float)
    N_BER     = 8_000
    h_sel_cir = _build_cir(H_SEL_DELAYS, H_SEL_GAINS, H_SEL_PHASES)

    ber_awgn_th  = _ber_awgn_theory(snr_range)
    ber_rayl_th  = _ber_rayleigh_theory(snr_range)

    # Rayleigh flat sim with CSI (BPSK: decision on real(rx * conj(h)/|h|))
    bits_tx_bpsk = np.random.randint(0, 2, N_BER).astype(np.uint8)
    syms_bpsk    = bits_to_symbols(bits_tx_bpsk, "BPSK")
    ber_rayl_sim = []
    ber_sel_sim  = []

    for snr_db in snr_range:
        # --- Rayleigh flat (with CSI equalisation) ---
        h_f = rayleigh_fading(len(syms_bpsk), 5, FS)
        rx_f = apply_fading(syms_bpsk, h_f)
        rx_f = awgn(rx_f, snr_db)
        # CSI equalize: multiply by conj(h)/|h|
        h_eq  = h_f[:len(rx_f)]
        rx_eq = rx_f * np.conj(h_eq) / (np.abs(h_eq) + 1e-12)
        bits_rx = symbols_to_bits(rx_eq, "BPSK")
        n = min(len(bits_tx_bpsk), len(bits_rx))
        ber_rayl_sim.append(np.mean(bits_tx_bpsk[:n] != bits_rx[:n]) + 1e-7)

        # --- Freq-selective (no equaliser) ---

        rx_s = fftconvolve(syms_bpsk, h_sel_cir)[:len(syms_bpsk)]
        rx_s = awgn(rx_s, snr_db)
        bits_rxs = symbols_to_bits(rx_s, "BPSK")
        ns = min(len(bits_tx_bpsk), len(bits_rxs))
        ber_sel_sim.append(np.mean(bits_tx_bpsk[:ns] != bits_rxs[:ns]) + 1e-7)

    ax.semilogy(snr_range, ber_awgn_th,  'k--',  linewidth=2,  label="AWGN (theory)")
    ax.semilogy(snr_range, ber_rayl_th,  'r--',  linewidth=2,  label="Rayleigh flat (theory)")
    ax.semilogy(snr_range, ber_rayl_sim, 'b-o',  ms=4,         label="Rayleigh flat (sim, CSI)")
    ax.semilogy(snr_range, ber_sel_sim,  'g-s',  ms=4,         label="Freq-selective (sim, no EQ)")
    ax.set_title("BER vs SNR — Effect of Fading")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.legend(fontsize=8)
    ax.grid(True, which='both')
    ax.set_ylim([1e-5, 0.5])

    # ── [1,0] OFDM vs Single-Carrier ─────────────────────────────────────────
    ax = axes[1, 0]
    snr_ofdm  = np.arange(0, 25, 2, dtype=float)
    N_FFT     = 64
    N_CP      = 16
    PILOT_IDX = np.linspace(2, 61, 8, dtype=int)
    DATA_IDX  = np.array([k for k in range(1, N_FFT - 1) if k not in PILOT_IDX])
    N_DATA    = len(DATA_IDX)
    N_SYMS    = 40    # OFDM symbols per SNR point

    h_sel_cir_arr = _build_cir(H_SEL_DELAYS, H_SEL_GAINS, H_SEL_PHASES)
    # Frequency-domain channel
    H_sel_fd = np.fft.fft(h_sel_cir_arr, n=N_FFT)

    ber_sc_sel   = []
    ber_ofdm_sel = []

    # Known pilot value (BPSK: +1)
    PILOT_VAL = 1.0 + 0j

    for snr_db in snr_ofdm:
        snr_lin = 10 ** (snr_db / 10)
        noise_std = np.sqrt(1 / (2 * snr_lin))

        # --- Single-carrier no EQ ---
        n_sc_bits = N_DATA * N_SYMS * 2  # QPSK → 2 bits, but we use BPSK here for comparison
        n_sc_bits = N_FFT * N_SYMS       # one BPSK per sample
        bits_sc   = np.random.randint(0, 2, n_sc_bits).astype(np.uint8)
        syms_sc   = bits_to_symbols(bits_sc, "BPSK")

        rx_sc = fftconvolve(syms_sc, h_sel_cir_arr)[:len(syms_sc)]
        rx_sc = awgn(rx_sc, snr_db)
        bits_sc_rx = symbols_to_bits(rx_sc, "BPSK")
        ns2 = min(len(bits_sc), len(bits_sc_rx))
        ber_sc_sel.append(np.mean(bits_sc[:ns2] != bits_sc_rx[:ns2]) + 1e-7)

        # --- OFDM + LS channel estimation ---
        all_bits_rx  = []
        all_bits_tx  = []

        for _ in range(N_SYMS):
            # Generate QPSK data bits (2 bits per subcarrier)
            n_data_bits = N_DATA * 2
            bits_d = np.random.randint(0, 2, n_data_bits).astype(np.uint8)
            syms_d = bits_to_symbols(bits_d, "QPSK")

            # Build frequency-domain OFDM symbol
            X = np.zeros(N_FFT, dtype=complex)
            X[PILOT_IDX] = PILOT_VAL
            X[DATA_IDX]  = syms_d[:N_DATA]

            # IFFT + CP
            x_td = np.fft.ifft(X)
            x_cp = np.concatenate([x_td[-N_CP:], x_td])

            # Channel: convolve with h_sel
            y_ch = fftconvolve(x_cp, h_sel_cir_arr)[:len(x_cp)]

            # Add AWGN
            noise = (noise_std * np.random.randn(len(y_ch))
                     + 1j * noise_std * np.random.randn(len(y_ch)))
            y_ch = y_ch + noise

            # Remove CP + FFT
            y_td = y_ch[N_CP: N_CP + N_FFT]
            Y    = np.fft.fft(y_td)

            # LS channel estimation at pilots
            H_est_pilots = Y[PILOT_IDX] / PILOT_VAL
            # Linear interpolation to all subcarriers
            H_est_all = (
                np.interp(np.arange(N_FFT), PILOT_IDX, np.real(H_est_pilots))
                + 1j * np.interp(np.arange(N_FFT), PILOT_IDX, np.imag(H_est_pilots))
            )

            # Equalise data subcarriers
            Y_data_eq = Y[DATA_IDX] / (H_est_all[DATA_IDX] + 1e-12)

            # Demodulate
            bits_rx_d = symbols_to_bits(Y_data_eq[:N_DATA], "QPSK")
            all_bits_rx.append(bits_rx_d[:n_data_bits])
            all_bits_tx.append(bits_d)

        bits_ofdm_tx = np.concatenate(all_bits_tx)
        bits_ofdm_rx = np.concatenate(all_bits_rx)
        nn = min(len(bits_ofdm_tx), len(bits_ofdm_rx))
        ber_ofdm_sel.append(np.mean(bits_ofdm_tx[:nn] != bits_ofdm_rx[:nn]) + 1e-7)

    ax.semilogy(snr_ofdm, ber_sc_sel,   'r-o', ms=4, label="SC (no EQ)")
    ax.semilogy(snr_ofdm, ber_ofdm_sel, 'b-s', ms=4, label="OFDM + LS est.")
    ax.semilogy(snr_ofdm, _ber_awgn_theory(snr_ofdm), 'k--', linewidth=1.5, label="AWGN theory (ref)")
    ax.set_title("OFDM vs Single-Carrier (Freq-Selective Channel)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.legend(fontsize=8)
    ax.grid(True, which='both')
    ax.set_ylim([1e-5, 1.0])

    # ── [1,1] Rician K-factor BER impact ─────────────────────────────────────
    ax = axes[1, 1]
    snr_ric   = np.arange(0, 21, 2, dtype=float)
    N_RIC     = 5_000
    K_db_ber  = [0, 3, 10, 20]

    bits_ric = np.random.randint(0, 2, N_RIC).astype(np.uint8)
    syms_ric = bits_to_symbols(bits_ric, "BPSK")

    for K_db in K_db_ber:
        K_lin  = 0.0 if K_db <= 0 else 10 ** (K_db / 10)
        lbl    = f"Rayleigh (K=−∞)" if K_db <= 0 else f"Rician K={K_db} dB"
        ber_k  = []
        for snr_db in snr_ric:
            h_k  = rician_fading(len(syms_ric), 5, FS, K_lin)
            rx_k = apply_fading(syms_ric, h_k)
            rx_k = awgn(rx_k, snr_db)
            # CSI equalize
            h_eq_k  = h_k[:len(rx_k)]
            rx_eq_k = rx_k * np.conj(h_eq_k) / (np.abs(h_eq_k) + 1e-12)
            bits_r  = symbols_to_bits(rx_eq_k, "BPSK")
            nk = min(len(bits_ric), len(bits_r))
            ber_k.append(np.mean(bits_ric[:nk] != bits_r[:nk]) + 1e-7)
        ax.semilogy(snr_ric, ber_k, '-o', ms=4, label=lbl)

    ax.semilogy(snr_ric, _ber_awgn_theory(snr_ric), 'k--', linewidth=2, label="AWGN (theory)")
    ax.set_title("BER vs SNR — Rician K-Factor")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.legend(fontsize=8)
    ax.grid(True, which='both')
    ax.set_ylim([1e-5, 0.5])

    # ── [1,2] Summary Text Panel ──────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis('off')

    mean_d_sel, rms_d_sel, Bc50_sel, Bc90_sel, ntaps_sel = stats_sel
    mean_d_flt, rms_d_flt, Bc50_flt, Bc90_flt, ntaps_flt = stats_flat

    T_c_slow_ms = T_c_slow * 1e3
    T_c_fast_ms = T_c_fast * 1e3

    summary = (
        "═══════════════════════════════════════════\n"
        "   CHANNEL CHARACTERISATION SUMMARY\n"
        "═══════════════════════════════════════════\n"
        "\n"
        "3-tap NLOS channel @ 1 MSPS:\n"
        f"  Mean excess delay : {mean_d_sel*1e6:6.2f} μs\n"
        f"  RMS delay spread  : {rms_d_sel*1e6:6.2f} μs\n"
        f"  B_c (50%)         : {Bc50_sel/1e3:6.1f} kHz\n"
        f"  B_c (90%)         : {Bc90_sel/1e3:6.1f} kHz\n"
        f"  Significant taps  : {ntaps_sel}\n"
        "\n"
        "Near-flat channel:\n"
        f"  Mean excess delay : {mean_d_flt*1e6:6.2f} μs\n"
        f"  RMS delay spread  : {rms_d_flt*1e6:6.2f} μs\n"
        f"  B_c (50%)         : {Bc50_flt/1e3:6.1f} kHz\n"
        f"  B_c (90%)         : {Bc90_flt/1e3:6.1f} kHz\n"
        f"  Significant taps  : {ntaps_flt}\n"
        "\n"
        "Time variation:\n"
        f"  f_d = 5 Hz   T_c = {T_c_slow_ms:6.1f} ms\n"
        f"  f_d = 200 Hz T_c = {T_c_fast_ms:6.2f} ms\n"
        "\n"
        "Classification (1 MSPS, f_d = 5 Hz):\n"
        f"  Sig. BW (1 MHz) >> B_c ({Bc50_sel/1e3:.0f} kHz)\n"
        f"  → FREQUENCY-SELECTIVE\n"
        f"  T_sym (1 μs) << T_c ({T_c_slow_ms:.0f} ms)\n"
        f"  → SLOW FADING\n"
        "  ┌─────────────────────────────────────┐\n"
        "  │ Regime: FREQ-SELECTIVE / SLOW FADING│\n"
        "  └─────────────────────────────────────┘\n"
    )

    ax.text(0.02, 0.98, summary, transform=ax.transAxes,
            fontsize=8.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f8f8', alpha=0.8))

    fig.tight_layout()
    out = "fading_doppler.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main simulation routine ───────────────────────────────────────────────────

def simulate(args):
    print("=== Channel Characterisation ===\n")

    # Build CIR vectors
    h_sel  = _build_cir(H_SEL_DELAYS,  H_SEL_GAINS,  H_SEL_PHASES)
    h_flat = _build_cir(H_FLAT_DELAYS, H_FLAT_GAINS, H_FLAT_PHASES)

    # PDP stats
    stats_sel  = pdp_stats(h_sel,  FS)
    stats_flat = pdp_stats(h_flat, FS)
    mean_d_sel, rms_d_sel, Bc50_sel, Bc90_sel, ntaps_sel = stats_sel
    mean_d_flt, rms_d_flt, Bc50_flt, Bc90_flt, ntaps_flt = stats_flat

    # Coherence time (Clarke's formula)
    T_c_slow = 0.423 / 5.0      # f_d = 5 Hz
    T_c_fast = 0.423 / 200.0    # f_d = 200 Hz

    # Print summary to console
    print("3-tap NLOS channel @ 1 MSPS:")
    print(f"  CIR: h = {np.round(h_sel, 4)}")
    print(f"  Mean excess delay    : {mean_d_sel*1e6:.2f} μs")
    print(f"  RMS delay spread σ_τ : {rms_d_sel*1e6:.2f} μs")
    print(f"  Coherence BW B_c(50%): {Bc50_sel/1e3:.1f} kHz")
    print(f"  Coherence BW B_c(90%): {Bc90_sel/1e3:.1f} kHz")
    print(f"  Significant taps     : {ntaps_sel}")
    print()
    print("Near-flat channel:")
    print(f"  CIR: h = {np.round(h_flat, 4)}")
    print(f"  Mean excess delay    : {mean_d_flt*1e6:.2f} μs")
    print(f"  RMS delay spread σ_τ : {rms_d_flt*1e6:.2f} μs")
    print(f"  Coherence BW B_c(50%): {Bc50_flt/1e3:.1f} kHz")
    print(f"  Coherence BW B_c(90%): {Bc90_flt/1e3:.1f} kHz")
    print(f"  Significant taps     : {ntaps_flt}")
    print()
    print("Time variation:")
    print(f"  f_d = 5 Hz:   T_c = {T_c_slow*1e3:.1f} ms  (Clarke: 0.423/f_d)")
    print(f"  f_d = 200 Hz: T_c = {T_c_fast*1e3:.2f} ms")
    print()
    print("Classification (1 MSPS signal, f_d = 5 Hz):")
    print(f"  Signal BW (1 MHz) >> B_c ({Bc50_sel/1e3:.0f} kHz)"
          "  → FREQUENCY-SELECTIVE")
    print(f"  Symbol period (1 μs) << T_c ({T_c_slow*1e3:.0f} ms)"
          " → SLOW FADING")
    print("  → Regime: FREQUENCY-SELECTIVE / SLOW")
    print()

    # Generate fading envelopes
    print("Generating fading envelopes ...")
    h_slow = rayleigh_fading(N, 5,   FS)
    h_fast = rayleigh_fading(N, 200, FS)

    # Plot Figure 1
    print("Plotting fading_structure.png ...")
    plot_structure(h_sel, h_flat, stats_sel, stats_flat)

    # Plot Figure 2
    print("Plotting fading_doppler.png ...")
    plot_doppler(h_slow, h_fast, Bc50_sel, T_c_slow, T_c_fast, stats_sel, stats_flat)

    print("\nDone.")


# ── OTA measurement routine ───────────────────────────────────────────────────

def ota_measure(args):
    """Transmit a CW tone and measure received envelope for live fading observation."""
    from common.pluto_config import connect_both
    tx, rx = connect_both(fc=args.fc, fs=1e6, gain_att=-40, gain_db=60)
    N_buf  = 2 ** 16

    # CW tone (complex, DC = constant envelope)
    cw = np.ones(N_buf, dtype=np.complex64) * 8192
    tx.tx(cw)

    print(f"[OTA] CW tone TX at {args.fc/1e6:.3f} MHz. Recording envelope ...")
    bufs = []
    for i in range(10):
        raw = rx.rx()
        iq  = raw.astype(np.complex64)
        bufs.append(iq)
        print(f"\r  Buffer {i+1}/10", end="", flush=True)
    tx.tx_destroy_buffer()
    print()

    iq_all = np.concatenate(bufs)
    env    = np.abs(iq_all)
    t_ms   = np.arange(len(env)) / 1e6 * 1e3

    plt.figure(figsize=(10, 4))
    plt.plot(t_ms, 20 * np.log10(env + 1e-6))
    plt.title("Live Received Envelope — OTA Fading Measurement")
    plt.xlabel("Time (ms)")
    plt.ylabel("Received Level (dBFS)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fading_ota.png", dpi=150)
    plt.close()
    print("  OTA fading plot: fading_ota.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Experiment 07 — Channel Modelling and Fading Demo"
    )
    p.add_argument("--mode", default="sim", choices=["sim", "ota"],
                   help="sim: pure simulation | ota: live measurement with Plutos")
    p.add_argument("--fc", type=float, default=915e6,
                   help="Carrier frequency (Hz), used in OTA mode only")
    args = p.parse_args()

    if args.mode == "sim":
        simulate(args)
    else:
        ota_measure(args)
