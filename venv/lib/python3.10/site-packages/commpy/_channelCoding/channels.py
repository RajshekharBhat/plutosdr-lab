"""Channel models for discrete and analog channels.

This module implements common channel models used in communications
simulations: Binary Symmetric Channel (BSC), Binary Erasure Channel (BEC),
and Additive White Gaussian Noise (AWGN) channel.

Functions are provided as static methods of the `Channels` class so callers
can use `Channels.bsc(...)`, `Channels.bec(...)`, and `Channels.awgn(...)`.
"""

from typing import Any

import numpy as np


class Channels:

    @staticmethod
    def bsc(bits: Any, p: float, rng: np.random.Generator | None = None) -> np.ndarray:
        """Binary Symmetric Channel (BSC).

        Flips each bit independently with probability `p`.

        Parameters
        - bits: array-like of 0/1 or boolean values.
        - p: flip probability in [0, 1].
        - rng: optional `np.random.Generator` for reproducibility.

        Returns:
        - ndarray of the same shape as `bits` containing the (possibly)
          flipped bits.
        """
        bits_arr = np.asarray(bits)
        if rng is None:
            rng = np.random.default_rng()
        flips = rng.random(bits_arr.shape) < float(p)

        # Preserve boolean dtype when possible
        if bits_arr.dtype == bool:
            return np.logical_xor(bits_arr, flips)

        # For numeric arrays (0/1), flip by doing 1 - bit where flips=True
        out = bits_arr.copy()
        out = np.where(flips, 1 - out, out)
        return out

    @staticmethod
    def bec(
        bits: Any,
        p: float,
        erasure_value: Any = -1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Binary Erasure Channel (BEC).

        Erases each symbol independently with probability `p`. Erased entries
        are replaced with `erasure_value` (default -1).

        Parameters
        - bits: array-like of symbols (commonly 0/1).
        - p: erasure probability in [0, 1].
        - erasure_value: value used to indicate erasure.
        - rng: optional `np.random.Generator` for reproducibility.

        Returns:
        - ndarray (dtype float) with erased entries set to `erasure_value`.
        """
        bits_arr = np.asarray(bits)
        if rng is None:
            rng = np.random.default_rng()
        erasures = rng.random(bits_arr.shape) < float(p)
        out = bits_arr.astype(float).copy()
        out[erasures] = erasure_value
        return out

    @staticmethod
    def awgn(x: Any, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
        """Additive White Gaussian Noise (AWGN) channel.

        Adds Gaussian noise to the input signal `x` such that the resulting
        signal-to-noise ratio (SNR) in dB is approximately `snr_db`.

        The function measures the average signal power as `mean(|x|^2)` and
        uses that to determine the noise power: noise_power = signal_power / snr_lin.

        For real-valued inputs, noise is real Gaussian with variance = noise_power.
        For complex-valued inputs, noise has independent real/imag parts each
        with variance = noise_power/2.

        Parameters
        - x: input signal (array-like, real or complex).
        - snr_db: desired SNR in dB (linear SNR = 10**(snr_db/10)).
        - rng: optional `np.random.Generator` for reproducibility.

        Returns:
        - noisy signal as ndarray with same shape and dtype as `x`.
        """
        x_arr = np.asarray(x)
        if rng is None:
            rng = np.random.default_rng()

        # average signal power per sample
        signal_power = float(np.mean(np.abs(x_arr)**2))
        if signal_power == 0:
            # avoid divide-by-zero; just add unit-variance noise scaled by snr
            signal_power = 1e-16

        snr_lin = 10.0**(float(snr_db) / 10.0)
        noise_power = signal_power / snr_lin

        if np.iscomplexobj(x_arr):
            sigma = np.sqrt(noise_power / 2.0)
            noise = sigma * (rng.normal(size=x_arr.shape) + 1j * rng.normal(size=x_arr.shape))
        else:
            sigma = np.sqrt(noise_power)
            noise = sigma * rng.normal(size=x_arr.shape)

        return x_arr + noise

    @staticmethod
    def rayleigh(x: Any, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
        """Rayleigh fading channel + AWGN.

        Parameters
        - x: Input signal (array-like, real or complex).
        - snr_db: Desired SNR in dB.
        - rng: Optional random generator.

        Returns:
        - Output after Rayleigh fading and AWGN.
        """
        x_arr = np.asarray(x)
        if rng is None:
            rng = np.random.default_rng()

        # Rayleigh coefficients (can be complex or real)
        if np.iscomplexobj(x_arr):
            h = (rng.normal(size=x_arr.shape) + 1j * rng.normal(size=x_arr.shape)) / np.sqrt(2)
        else:
            h = rng.rayleigh(scale=1.0, size=x_arr.shape)

        faded = x_arr * h
        return Channels.awgn(faded, snr_db, rng)

    @staticmethod
    def rician(
        x: Any,
        snr_db: float,
        K: float = 10.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Rician fading channel + AWGN.

        Parameters
        - x: Input signal (array-like, real or complex).
        - snr_db: desired SNR in dB.
        - K: Rician K-factor (power ratio of LOS to scattered).
        - rng: Optional random generator.

        Returns:
        - Output after Rician fading and AWGN.
        """
        x_arr = np.asarray(x)
        if rng is None:
            rng = np.random.default_rng()

        # Rician fading: specular (LOS) + scattered
        # K-factor: ratio of direct/single path to scatter power (linear)
        K_lin = float(K)
        s = np.sqrt(K_lin / (K_lin + 1))
        sigma = np.sqrt(1 / (2 * (K_lin + 1)))

        if np.iscomplexobj(x_arr):
            h = s + sigma * (rng.normal(size=x_arr.shape) + 1j * rng.normal(size=x_arr.shape))
        else:
            h = s + sigma * rng.normal(size=x_arr.shape)

        faded = x_arr * h
        return Channels.awgn(faded, snr_db, rng)

    @staticmethod
    def z_channel(bits: Any, p: float, rng: np.random.Generator | None = None) -> np.ndarray:
        """Z-Channel.

        Flips 1 to 0 with probability p. 0 stays 0 always.

        Parameters
        - bits: array-like (0/1 or bool)
        - p: error probability for 1→0.
        - rng: random generator.

        Returns:
        - ndarray of same shape.
        """
        bits_arr = np.asarray(bits)
        if rng is None:
            rng = np.random.default_rng()
        flips = (bits_arr == 1) & (rng.random(bits_arr.shape) < p)
        out = bits_arr.copy()
        out[flips] = 0
        return out

    @staticmethod
    def gilbert_elliott(
        bits: Any,
        p_gb: float,
        p_bg: float,
        p_good: float = 0.0,
        p_bad: float = 0.2,
        rng: np.random.Generator | None = None,
        init_state: str = 'good',
    ) -> np.ndarray:
        """Gilbert-Elliott two-state bursty channel.

        Parameters
        - bits: input bits array-like (0/1 or bool)
        - p_gb: prob. to switch Good->Bad
        - p_bg: prob. to switch Bad->Good
        - p_good: error of 'good' state
        - p_bad: error of 'bad' state
        - rng: random number generator
        - init_state: "good" or "bad"

        Returns:
        - output bits (ndarray)
        """
        bits_arr = np.asarray(bits)
        if rng is None:
            rng = np.random.default_rng()
        N = bits_arr.size

        state = 0 if init_state == 'good' else 1
        states = np.zeros(N, dtype=int)
        for i in range(1, N):
            if state == 0 and rng.random() < p_gb:
                state = 1
            elif state == 1 and rng.random() < p_bg:
                state = 0
            states[i] = state

        errors = np.where(states == 0, rng.random(N) < p_good, rng.random(N) < p_bad)
        if bits_arr.dtype == bool:
            return np.logical_xor(bits_arr, errors)
        out = bits_arr.copy()
        out = np.where(errors, 1 - out, out)
        return out

    @staticmethod
    def quantize(
        x: Any,
        bits: int = 8,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> np.ndarray:
        """Uniform quantization of an analog signal.

        Parameters
        - x: input real signal
        - bits: number of quantization bits
        - vmin, vmax: optional min/max for clipping range

        Returns:
        - quantized signal
        """
        x_arr = np.asarray(x)
        if vmin is None:
            vmin = np.min(x_arr)
        if vmax is None:
            vmax = np.max(x_arr)
        levels = 2**bits
        x_clipped = np.clip(x_arr, vmin, vmax)
        q = np.round((x_clipped - vmin) / (vmax - vmin) * (levels - 1))
        xq = vmin + q / (levels - 1) * (vmax - vmin)
        return xq


__all__ = ['Channels']
