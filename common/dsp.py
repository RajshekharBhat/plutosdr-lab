"""
DSP building blocks:
  - AGC (automatic gain control)
  - Timing recovery (Gardner TED)
  - Carrier frequency offset correction (frequency-locked loop)
  - Phase recovery (Costas loop / decision-directed PLL)
  - DC offset removal
  - SNR estimation (M2M4 moment-based)
"""

import numpy as np
from scipy.signal import lfilter


# ---------- DC removal ----------

def remove_dc(x: np.ndarray, alpha: float = 0.999) -> np.ndarray:
    """IIR DC blocking filter."""
    b = [1.0, -1.0]
    a = [1.0, -alpha]
    return lfilter(b, a, x)


# ---------- AGC ----------

def agc(x: np.ndarray, target_power: float = 1.0, alpha: float = 0.01) -> np.ndarray:
    """Simple exponential-averaging AGC. Returns gain-adjusted signal."""
    out   = np.zeros_like(x)
    gain  = 1.0
    p_est = 1.0
    for i, s in enumerate(x):
        out[i] = s * gain
        p_est  = (1 - alpha) * p_est + alpha * abs(out[i]) ** 2
        gain   = np.sqrt(target_power / (p_est + 1e-12))
    return out


# ---------- Frequency offset correction ----------

def correct_freq_offset(x: np.ndarray, freq_offset_hz: float, fs: float) -> np.ndarray:
    """Multiply x by a complex exponential to remove freq_offset_hz."""
    t = np.arange(len(x)) / fs
    return x * np.exp(-1j * 2 * np.pi * freq_offset_hz * t)


class FreqLockLoop:
    """
    Second-order frequency-locked loop for residual CFO tracking.
    Suitable after coarse correction. Uses non-data-aided error detector.
    """
    def __init__(self, fs: float, bw_norm: float = 0.01):
        # Loop filter coefficients (Proakis/Savaux design)
        damp  = 1.0 / np.sqrt(2)
        wn    = 2 * np.pi * bw_norm
        self.K1   = 2 * damp * wn
        self.K2   = wn ** 2
        self._freq = 0.0
        self._phase = 0.0
        self._int   = 0.0
        self._fs    = fs

    def step(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        for i, s in enumerate(x):
            s_corr          = s * np.exp(-1j * self._phase)
            out[i]          = s_corr
            err             = np.real(s_corr) * np.imag(s_corr)   # non-data-aided
            self._int      += self.K2 * err
            self._freq      = self.K1 * err + self._int
            self._phase    += self._freq
        return out


# ---------- Timing recovery (Gardner TED) ----------

def gardner_timing_recovery(x: np.ndarray, sps: int,
                             mu0: float = 0.0,
                             loop_bw: float = 0.01) -> np.ndarray:
    """
    Gardner timing error detector with linear interpolation.
    sps : samples per symbol (integer nominal).
    Returns (symbols, mu_trace) where mu_trace is the timing offset trajectory.
    """
    N     = len(x)
    out   = []
    mu    = mu0          # fractional timing offset [0, 1)
    cnt   = sps          # counter until next sample decision

    K1    = 2 * loop_bw
    K2    = loop_bw ** 2
    integ = 0.0

    prev  = x[0]
    mid   = x[sps // 2]
    i     = sps

    while i < N - sps:
        # Linear interpolation
        frac = mu - int(mu)
        s    = x[i] * (1 - frac) + x[i + 1] * frac if frac > 1e-6 else x[i]

        out.append(s)

        # Gardner error: e = Re[ (x[k] - x[k-1]) * conj(x[k-0.5]) ]
        mid_i   = i - sps // 2
        mid_s   = x[mid_i]
        err     = np.real((s - prev) * np.conj(mid_s))

        integ  += K2 * err
        mu     += K1 * err + integ

        # Advance by nominal sps + fractional correction
        step    = sps + int(np.round(mu))
        mu     -= int(np.round(mu))
        prev    = s
        i      += step

    return np.array(out)


# ---------- Phase recovery (Costas loop) ----------

class CostasLoop:
    """
    Decision-directed Costas loop for phase and residual frequency tracking.
    Supports BPSK and QPSK.
    """
    def __init__(self, order: int = 2, bw_norm: float = 0.005):
        self._order  = order        # 2=BPSK, 4=QPSK
        damp  = 1.0 / np.sqrt(2)
        wn    = 2 * np.pi * bw_norm
        self.K1     = 2 * damp * wn
        self.K2     = wn ** 2
        self._phase = 0.0
        self._freq  = 0.0
        self._int   = 0.0

    def _error(self, s: complex) -> float:
        if self._order == 2:
            return np.real(s) * np.imag(s)
        else:
            # QPSK: decision-directed
            d = (np.sign(np.real(s)) + 1j * np.sign(np.imag(s))) / np.sqrt(2)
            return np.imag(s * np.conj(d))

    def step(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        for i, s in enumerate(x):
            s_corr       = s * np.exp(-1j * self._phase)
            out[i]       = s_corr
            err          = self._error(s_corr)
            self._int   += self.K2 * err
            self._freq   = self.K1 * err + self._int
            self._phase += self._freq
        return out


# ---------- SNR estimation (M2M4 moment-based) ----------

def estimate_snr_m2m4(x: np.ndarray) -> float:
    """
    Non-data-aided SNR estimation using 2nd and 4th moments.
    Works for PSK/QAM. Returns SNR in dB.
    """
    M2 = np.mean(np.abs(x) ** 2)
    M4 = np.mean(np.abs(x) ** 4)
    # kurtosis-based: SNR = M2^2 / (M4 - M2^2)  (approx for AWGN+PSK)
    denom = M4 - M2 ** 2
    if denom < 1e-12:
        return 40.0   # very high SNR
    snr_lin = M2 ** 2 / denom
    return 10 * np.log10(max(snr_lin, 1e-6))
