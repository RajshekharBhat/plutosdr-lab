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
    Vectorised: builds (N_osc × n_samples) phase matrix in one shot.
    Returns complex envelope h[n] of length n_samples.
    """
    N_osc = 32
    theta = np.random.uniform(0, 2 * np.pi, N_osc)          # (N_osc,)
    phi   = np.random.uniform(0, 2 * np.pi, N_osc)          # (N_osc,)
    t     = np.arange(n_samples) / fs                       # (N,)
    alpha = 2 * np.pi * np.arange(N_osc) / N_osc + theta   # (N_osc,)
    # Doppler frequency per oscillator: (N_osc,)
    f_i   = doppler_hz * np.cos(alpha)
    f_q   = doppler_hz * np.sin(alpha)
    # Phase matrices: (N_osc, N) via broadcasting
    ph_i  = 2 * np.pi * f_i[:, None] * t[None, :] + phi[:, None]
    ph_q  = 2 * np.pi * f_q[:, None] * t[None, :] + phi[:, None]
    h_i   = np.sum(np.cos(ph_i), axis=0)                    # (N,)
    h_q   = np.sum(np.cos(ph_q), axis=0)
    return (h_i + 1j * h_q) / np.sqrt(2 * N_osc)


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


# ---------- Rician fading ----------

def rician_fading(n_samples: int, doppler_hz: float, fs: float,
                  K_linear: float = 0.0) -> np.ndarray:
    """
    Generate Rician flat-fading envelope (sum-of-sinusoids + LOS).
    K_linear = 0        → Rayleigh (no LOS).
    K_linear > 0        → Rician; K = P_LOS / P_scatter.
    Total power is normalised to 1.
    """
    scatter = rayleigh_fading(n_samples, doppler_hz, fs)
    # Normalise scatter to unit power
    rms = np.sqrt(np.mean(np.abs(scatter) ** 2)) + 1e-12
    scatter = scatter / rms
    # Scale scatter power to 1/(K+1)
    scatter_scaled = scatter / np.sqrt(K_linear + 1)
    if K_linear <= 0.0:
        return scatter_scaled
    # LOS component: power = K/(K+1), random initial phase
    phi_los = np.random.uniform(0, 2 * np.pi)
    los = np.sqrt(K_linear / (K_linear + 1)) * np.exp(1j * phi_los)
    return scatter_scaled + los


# ---------- PDP statistics ----------

def pdp_stats(h_td: np.ndarray, fs: float):
    """
    Compute PDP-derived channel statistics from a CIR vector h_td.

    Returns
    -------
    mean_delay_s  : mean excess delay  (seconds)
    rms_delay_s   : RMS delay spread σ_τ  (seconds)
    B_c_50_hz     : coherence bandwidth (50% criterion)  ≈ 1/(5 σ_τ)
    B_c_90_hz     : coherence bandwidth (90% criterion)  ≈ 1/(2π σ_τ)
    n_sig_taps    : number of taps above 1% of peak power
    """
    pdp = np.abs(h_td) ** 2
    total = pdp.sum()
    if total == 0.0:
        return 0.0, 0.0, np.inf, np.inf, 0
    pdp_n  = pdp / total
    delays = np.arange(len(h_td)) / fs          # seconds
    mean_d = float(np.dot(delays, pdp_n))
    rms_d  = float(np.sqrt(np.dot((delays - mean_d) ** 2, pdp_n)))
    B_50   = 1.0 / (5.0 * rms_d)      if rms_d > 0 else np.inf
    B_90   = 1.0 / (2 * np.pi * rms_d) if rms_d > 0 else np.inf
    n_taps = int(np.sum(pdp > pdp.max() * 0.01))
    return mean_d, rms_d, B_50, B_90, n_taps


# ---------- Jake's Doppler PSD ----------

def jakes_doppler_psd(f_range: np.ndarray, f_d: float) -> np.ndarray:
    """
    Jake's theoretical U-shaped Doppler PSD.
    S(f) = 1 / (π f_d √(1 - (f/f_d)²))   for |f| < f_d
    Returns zero outside the Doppler bandwidth.
    """
    psd  = np.zeros_like(f_range, dtype=float)
    mask = np.abs(f_range) < f_d * 0.999
    psd[mask] = 1.0 / (np.pi * f_d * np.sqrt(1.0 - (f_range[mask] / f_d) ** 2))
    return psd
