"""IQWaveform class for generating and plotting IQ modulated waveforms."""

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np


class IQWaveform:
    """Class for generating and plotting IQ modulated waveforms."""

    def __init__(
        self,
        I: Sequence[float],             # or np.ndarray
        Q: Sequence[float],             # or np.ndarray
        T: float,
        fs: float,
        f0: float = 0.0,
        pulse_shape: Callable[[np.ndarray], np.ndarray] | None = None,
        span: int = 4,
    ) -> None:
        """Parameters:
        - I, Q: arrays of symbols (real numbers, usually from modulation, e.g.
            I = syms.real, Q = syms.imag)
        - T: symbol period (seconds)
        - fs: sample rate (Hz)
        - f0: carrier frequency (Hz). If 0, output baseband only.
        - pulse_shape: function(tau) returning g_s(tau). Default: rect pulse.
        - span: Pulse truncation window in multiples of T (for computational
            efficiency; e.g. 4)
        """
        self.I = np.asarray(I)
        self.Q = np.asarray(Q)
        self.T = T
        self.fs = fs
        self.f0 = f0
        self.span = span
        self.N = len(self.I)
        if pulse_shape is None:
            self.pulse_shape = lambda tau: ((tau >= 0) & (tau < T)).astype(float)
        else:
            self.pulse_shape = pulse_shape

        self.t, self.s, self.s_I, self.s_Q = self._make_waveform()

    def _make_waveform(self):
        N = self.N
        T = self.T
        fs = self.fs
        span_window = self.span

        L = int(N * T * fs)
        t = np.arange(L) / fs

        s_I = np.zeros_like(t)
        s_Q = np.zeros_like(t)

        # Synthesizing S_I(t) and S_Q(t) as shaped pulse trains
        for n, (I_n, Q_n) in enumerate(zip(self.I, self.Q)):
            tn = n * T
            # only non-zero within [tn, tn+span*T]
            mask = (t >= tn) & (t < tn + span_window * T)
            tau = t[mask] - tn
            gs = self.pulse_shape(tau)
            s_I[mask] += I_n * gs
            s_Q[mask] += Q_n * gs

        # Modulate to bandpass, as per your formula:
        carrier_cos = np.sqrt(2) * np.cos(2 * np.pi * self.f0 * t)
        carrier_sin = np.sqrt(2) * np.sin(2 * np.pi * self.f0 * t)
        s = s_I * carrier_cos - s_Q * carrier_sin
        return t, s, s_I, s_Q

    def plot_waveform(self):
        """Plot the analog transmit signal \tilde{S}(t)."""
        plt.figure(figsize=(10, 3))
        plt.plot(self.t, self.s, lw=1.2)
        plt.xlabel('t [s]')
        plt.ylabel('$\\tilde{S}(t)$')
        plt.title('Bandpass IQ Modulated Signal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_IQ_baseband(self):
        """Plot baseband I/Q components as a function of time."""
        plt.figure(figsize=(10, 4))
        plt.plot(self.t, self.s_I, label='I(t)', lw=1.2)
        plt.plot(self.t, self.s_Q, label='Q(t)', lw=1.2)
        plt.xlabel('t [s]')
        plt.ylabel('Amplitude')
        plt.title('Baseband I/Q Components (Pulse Shaped)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_eye(self, N=2):
        """Plot eye diagram for I(t)."""
        period = int(self.T * self.fs)
        plt.figure()
        for k in range(len(self.s_I) // period - N):
            plt.plot(np.arange(period) / self.fs, self.s_I[k * period:(k + 1) * period], 'b')
        plt.xlabel('t [symbol period]')
        plt.title('Eye diagram (I)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # You can add spectrum plotting or save-to-file as needed.


# %%
# Example usage:
if __name__ == '__main__':
    from CommPy.channelCoding.modulation import OOK
    bits = np.random.randint(0, 2, 16)
    syms = OOK.modulate(bits)
    I = syms.real
    Q = syms.imag
    T = 1e-3
    fs = 300000
    f0 = 20000

    tx = IQWaveform(I, Q, T, fs, f0)  # Default: rectangular pulse
    tx.plot_waveform()
    tx.plot_IQ_baseband()
    tx.plot_eye()
