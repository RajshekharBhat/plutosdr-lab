#!/usr/bin/env python3
"""
OFDM Spectrum — Educational Plot
=================================
Shows:
  1. Individual subcarrier spectra (sinc functions) and orthogonality
  2. Overall OFDM spectrum with null subcarriers annotated

All axes in normalised frequency (f / delta_f), so the plot is valid
for any bandwidth B.  Set B below to scale the Hz axis.

Usage:
    python plot_ofdm_spectrum.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Parameters ────────────────────────────────────────────────────────────────
B  = 2e6          # total available bandwidth (Hz) — change freely
N  = 64           # FFT size
N_CP = 16

delta_f = B / N   # subcarrier spacing  (31.25 kHz at B=2 MHz)
T       = 1 / delta_f   # useful symbol duration = N/B  (32 µs at B=2 MHz)

# Subcarrier map (matching ofdm_tx.py)
PILOT_IDX = np.array([7, 21, 43, 57])
NULL_IDX  = np.concatenate([np.arange(27, 38), [0]])
DATA_IDX  = np.array([i for i in range(N)
                       if i not in PILOT_IDX and i not in NULL_IDX])
ACTIVE_IDX = np.sort(np.concatenate([DATA_IDX, PILOT_IDX]))

# ── Frequency axis (dense, in normalised units k = f/delta_f) ─────────────────
k_axis = np.linspace(-3, N + 3, 80000)   # subcarrier-index units

# ── Sinc for subcarrier n: sinc((k - n)) ─────────────────────────────────────
# np.sinc(x) = sin(pi x)/(pi x)  — normalised sinc
def sc_spectrum(k, n):
    return np.sinc(k - n)

# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                         gridspec_kw={'hspace': 0.45})

# ══════════════════════════════════════════════════════════════════════════════
# Panel 1 — Individual subcarrier spectra
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]
SHOW = [5, 6, 7, 8, 9, 10]
cmap  = plt.cm.tab10(np.linspace(0, 0.6, len(SHOW)))
zoom  = np.linspace(SHOW[0] - 2.5, SHOW[-1] + 2.5, 20000)

for i, n in enumerate(SHOW):
    s = sc_spectrum(zoom, n)
    ax.plot(zoom * delta_f / 1e3, s, color=cmap[i], lw=1.4,
            label=f'$k={n}$')
    # mark peak
    ax.plot(n * delta_f / 1e3, 1.0, 'o', color=cmap[i], ms=5, zorder=6)

# vertical lines at subcarrier positions — show zero crossings
for n in SHOW:
    ax.axvline(n * delta_f / 1e3, color='grey', lw=0.6, ls=':', alpha=0.6)

ax.axhline(0, color='k', lw=0.6)

# Annotate Δf
n0, n1 = SHOW[1], SHOW[2]
y_ann = -0.22
ax.annotate('', xy=(n1 * delta_f / 1e3, y_ann),
            xytext=(n0 * delta_f / 1e3, y_ann),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
ax.text((n0 + n1) / 2 * delta_f / 1e3, y_ann - 0.06,
        r'$\Delta f = B/N$', ha='center', fontsize=9)

# Annotate orthogonality
ax.annotate('zero at every\nneighbouring subcarrier\n(orthogonality)',
            xy=(SHOW[2] * delta_f / 1e3, 0.0),
            xytext=((SHOW[3] + 1) * delta_f / 1e3, 0.55),
            arrowprops=dict(arrowstyle='->', color='dimgray', lw=1),
            fontsize=8, color='dimgray')

ax.set_xlim([zoom[0] * delta_f / 1e3, zoom[-1] * delta_f / 1e3])
ax.set_ylim([-0.35, 1.25])
ax.set_xlabel('Frequency  (kHz)', fontsize=10)
ax.set_ylabel('Amplitude', fontsize=10)
ax.set_title('Individual subcarrier spectra  —  each is a $\\mathrm{sinc}$ '
             'with zeros at all other subcarrier frequencies',
             fontsize=10)
ax.legend(ncol=len(SHOW), fontsize=8, loc='upper right')
ax.grid(True, alpha=0.25)

# ══════════════════════════════════════════════════════════════════════════════
# Panel 2 — Overall OFDM power spectral density
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]

# PSD = sum of sinc^2 over active subcarriers (equal-power, independent phases)
psd = np.zeros(len(k_axis))
for n in ACTIVE_IDX:
    psd += sc_spectrum(k_axis, n) ** 2

psd_db = 10 * np.log10(psd / psd.max() + 1e-9)
freq_khz = k_axis * delta_f / 1e3

ax.plot(freq_khz, psd_db, 'steelblue', lw=1.4, label='OFDM PSD (52 active subcarriers)')

# ── Shade null regions ────────────────────────────────────────────────────────
# DC
ax.axvspan((-0.5) * delta_f / 1e3,
           ( 0.5) * delta_f / 1e3,
           alpha=0.25, color='red', zorder=2)
# Guard band (27–37)
ax.axvspan(26.5 * delta_f / 1e3,
           37.5 * delta_f / 1e3,
           alpha=0.20, color='orange', zorder=2)

# ── Reference lines ───────────────────────────────────────────────────────────
ax.axvline(0,         color='red',    ls='--', lw=1.0, label='DC  (bin 0)')
ax.axvline(32 * delta_f / 1e3,
           color='dimgray', ls='--', lw=1.0, label=r'Nyquist  $F_s/2$')

# ── Annotations ──────────────────────────────────────────────────────────────
# Guard band arrow
ax.annotate('Guard band\nbins 27–37\n(straddles Nyquist)',
            xy   =(32 * delta_f / 1e3, -8),
            xytext=((32 + 8) * delta_f / 1e3, -20),
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.2),
            fontsize=8, color='darkorange')

# DC annotation
ax.annotate('DC null\n(LO leakage)',
            xy   =(0, -8),
            xytext=(5 * delta_f / 1e3, -22),
            arrowprops=dict(arrowstyle='->', color='crimson', lw=1.2),
            fontsize=8, color='crimson')

# Active bandwidth arrow
f_lo = ACTIVE_IDX[ 0] * delta_f / 1e3
f_hi = ACTIVE_IDX[-1] * delta_f / 1e3
ax.annotate('', xy=(f_hi, -33), xytext=(f_lo, -33),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.3))
ax.text((f_lo + f_hi) / 2, -36,
        rf'Active BW $= {len(ACTIVE_IDX)}/{N} \times B$',
        ha='center', fontsize=9)

# Total bandwidth arrow
ax.annotate('', xy=(N * delta_f / 1e3, -28),
            xytext=(0, -28),
            arrowprops=dict(arrowstyle='<->', color='steelblue', lw=1.3))
ax.text(N / 2 * delta_f / 1e3, -31, r'$B = N \cdot \Delta f$',
        ha='center', fontsize=9, color='steelblue')

# Δf label on one subcarrier
ax.annotate('',
            xy   =(( 2)   * delta_f / 1e3, -3),
            xytext=((3)   * delta_f / 1e3, -3),
            arrowprops=dict(arrowstyle='<->', color='k', lw=1))
ax.text(2.5 * delta_f / 1e3, -5.5,
        r'$\Delta f$', ha='center', fontsize=9)

# ── Legend patches ────────────────────────────────────────────────────────────
null_patch  = mpatches.Patch(color='red',    alpha=0.4, label='DC null')
guard_patch = mpatches.Patch(color='orange', alpha=0.4, label='Guard band')
ax.legend(handles=[
    plt.Line2D([0],[0], color='steelblue', lw=1.4, label='OFDM PSD (52 active)'),
    null_patch, guard_patch,
    plt.Line2D([0],[0], color='red',     ls='--', lw=1, label='DC (bin 0)'),
    plt.Line2D([0],[0], color='dimgray', ls='--', lw=1, label=r'Nyquist $F_s/2$'),
], fontsize=8, loc='lower right')

ax.set_xlim([(-3) * delta_f / 1e3, (N + 3) * delta_f / 1e3])
ax.set_ylim([-42, 8])
ax.set_xlabel('Frequency  (kHz)', fontsize=10)
ax.set_ylabel('Power  (dB, normalised)', fontsize=10)
ax.set_title(f'Overall OFDM spectrum  —  $N={N}$ subcarriers, '
             f'$B={B/1e6:.0f}$ MHz,  $\\Delta f = B/N = {delta_f/1e3:.2f}$ kHz',
             fontsize=10)
ax.grid(True, alpha=0.25)

# ── Save ─────────────────────────────────────────────────────────────────────
plt.savefig('ofdm_spectrum.png', dpi=150, bbox_inches='tight')
print(f'Saved: ofdm_spectrum.png')
print(f'B = {B/1e6:.1f} MHz   delta_f = {delta_f/1e3:.3f} kHz   '
      f'T = {T*1e6:.1f} us   Active BW = {len(ACTIVE_IDX)/N*B/1e6:.3f} MHz')
plt.show()
