"""
Channel models for simulation / over-the-air impairment characterization.

  - AWGN
  - Frequency-selective multipath (FIR tap model)
  - Rayleigh / Rician flat fading
  - Jake's Doppler spectrum fading (time-varying)
  - Channel estimation: LS and MMSE (for OFDM pilot grids)
"""

import numpy as np
from scipy.signal import fftconvolve


# ---------- AWGN ----------

def awgn(x: np.ndarray, snr_db: float) -> np.ndarray:
    """Add complex AWGN to signal x at given SNR (dB) relative to signal power."""
    p_sig  = np.mean(np.abs(x) ** 2)
    snr    = 10 ** (snr_db / 10)
    p_noise = p_sig / snr
    noise  = np.sqrt(p_noise / 2) * (np.random.randn(len(x)) + 1j * np.random.randn(len(x)))
    return x + noise


# ---------- Multipath (frequency-selective) ----------

def multipath_channel(x: np.ndarray, delays_samples: list, gains_db: list,
                      phases_deg: list = None) -> np.ndarray:
    """
    Apply a static multipath channel.
    delays_samples : list of tap delays in samples
    gains_db       : list of tap gains in dB (relative to first tap)
    phases_deg     : list of tap phases in degrees (random if None)
    Returns convolved signal.
    """
    max_delay = max(delays_samples)
    h = np.zeros(max_delay + 1, dtype=complex)
    if phases_deg is None:
        phases_deg = np.random.uniform(0, 360, len(delays_samples))
    for d, g, p in zip(delays_samples, gains_db, phases_deg):
        h[d] += 10 ** (g / 20) * np.exp(1j * np.deg2rad(p))
    return fftconvolve(x, h)[:len(x)]


def example_multipath_channel():
    """Return (delays, gains, phases) for a 3-tap indoor channel example."""
    delays = [0,  5, 12]        # samples (at 1 MSPS → 0, 5, 12 μs)
    gains  = [0, -6, -12]       # dB
    phases = [0, 45, 120]       # deg
    return delays, gains, phases


# ---------- Flat Rayleigh / Rician fading ----------

def rayleigh_fading(n_samples: int, doppler_hz: float, fs: float) -> np.ndarray:
    """
    Generate a Rayleigh flat-fading envelope using sum-of-sinusoids (Jake's model).
    Returns complex envelope h[n] of length n_samples.
    """
    N_osc = 32
    theta = np.random.uniform(0, 2 * np.pi, N_osc)
    phi   = np.random.uniform(0, 2 * np.pi, N_osc)
    f_d   = doppler_hz
    t     = np.arange(n_samples) / fs
    h_i   = sum(np.cos(2 * np.pi * f_d * np.cos(2 * np.pi * k / N_osc + theta[k]) * t + phi[k])
                for k in range(N_osc))
    h_q   = sum(np.cos(2 * np.pi * f_d * np.sin(2 * np.pi * k / N_osc + theta[k]) * t + phi[k])
                for k in range(N_osc))
    h     = (h_i + 1j * h_q) / np.sqrt(2 * N_osc)
    return h


def apply_fading(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply flat-fading envelope h (same length as x) to signal x."""
    return x * h[:len(x)]


# ---------- Channel estimation (OFDM pilot-based) ----------

def ls_channel_estimate(Y_pilots: np.ndarray, X_pilots: np.ndarray) -> np.ndarray:
    """
    Least-squares channel estimate at pilot subcarriers.
    Y_pilots : received pilot symbols
    X_pilots : known transmitted pilot symbols
    Returns H_est at pilot positions.
    """
    return Y_pilots / X_pilots


def interpolate_channel(H_pilots: np.ndarray, pilot_indices: np.ndarray,
                         N_fft: int) -> np.ndarray:
    """
    Linear interpolation of channel from pilot subcarriers to all subcarriers.
    """
    all_k = np.arange(N_fft)
    H_all = np.interp(all_k, pilot_indices,
                      np.real(H_pilots)) + \
            1j * np.interp(all_k, pilot_indices, np.imag(H_pilots))
    return H_all


def mmse_channel_estimate(Y_pilots: np.ndarray, X_pilots: np.ndarray,
                           R_hh: np.ndarray, snr_lin: float) -> np.ndarray:
    """
    MMSE channel estimation at pilot subcarriers.
    R_hh     : channel covariance matrix (N_pilot x N_pilot)
    snr_lin  : linear SNR estimate
    Returns MMSE estimate H_mmse.
    """
    H_ls  = ls_channel_estimate(Y_pilots, X_pilots)
    N     = len(H_ls)
    W     = R_hh @ np.linalg.inv(R_hh + (1 / snr_lin) * np.eye(N))
    return W @ H_ls
