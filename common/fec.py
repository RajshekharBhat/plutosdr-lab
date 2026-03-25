"""
Forward Error Correction (FEC) — Convolutional codes.

Implements a pure-NumPy rate-1/n convolutional encoder and hard-decision
Viterbi decoder.  No external FEC library required.

Generator polynomial convention (same as MATLAB / commpy / ITU-T):
  - Polynomial given as an integer; bit k (from LSB) = 1 means include
    the shift-register tap delayed by k steps.
  - Bit 0  (LSB) of polynomial → current input  x_t
  - Bit M  (MSB) of polynomial → oldest stored input x_{t-M}
  - e.g. rate-1/2, K=7, G=[0o133, 0o171] is the NASA standard code.

Usage example:
    from common.fec import ConvCode
    code = ConvCode([0o133, 0o171], K=7)   # rate 1/2, K=7
    enc  = code.encode(info_bits)           # adds M=6 tail zeros
    dec  = code.decode(enc, len(info_bits)) # hard-decision Viterbi
"""

import numpy as np


class ConvCode:
    """Rate-1/n convolutional encoder + hard-decision Viterbi decoder."""

    def __init__(self, generators: list, K: int):
        """
        Parameters
        ----------
        generators : list of int
            Generator polynomials (integers, octal notation common).
            len(generators) = n  →  rate = 1/n.
        K : int
            Constraint length (total shift-register length including input).
            Memory M = K − 1.
        """
        self.K = K
        self.M = K - 1
        self.n = len(generators)       # output bits per input bit
        self.rate = 1.0 / self.n
        self.generators = generators
        self.n_states = 1 << self.M    # 2^M

        self._next  = np.empty((self.n_states, 2), dtype=np.int32)
        self._out   = np.empty((self.n_states, 2), dtype=np.int32)
        self._build_tables()

    # ──────────────────────────────────────────────────────────
    #  Internal trellis table
    # ──────────────────────────────────────────────────────────

    def _build_tables(self):
        """Pre-compute next-state, output, and reverse-predecessor tables."""
        M = self.M
        n = self.n
        for s in range(self.n_states):
            for b in range(2):
                # Shift register: state s = [x_{t-1} ... x_{t-M}] (MSB = newest)
                # After receiving input b = x_t:
                #   new state = (b << (M-1)) | (s >> 1)   (shift out oldest bit)
                ns = ((b << (M - 1)) | (s >> 1)) if M > 0 else 0
                self._next[s, b] = ns

                # Full K-bit register: MSB = current input, LSB = oldest
                reg = (b << M) | s
                packed_out = 0
                for j, g in enumerate(self.generators):
                    bit = bin(g & reg).count("1") & 1   # parity
                    packed_out |= bit << (n - 1 - j)
                self._out[s, b] = packed_out

        # Precompute expected output bit arrays: _exp[s, b, j] ∈ {0, 1}
        self._exp = np.empty((self.n_states, 2, n), dtype=np.uint8)
        for s in range(self.n_states):
            for b in range(2):
                out = int(self._out[s, b])
                for j in range(n):
                    self._exp[s, b, j] = (out >> (n - 1 - j)) & 1

        # Reverse lookup: for each next-state ns, its 2 predecessor (prev_s, prev_b) pairs.
        # For all rate-1/n codes (k=1), each state has exactly 2 predecessors.
        self._pred_s = np.zeros((self.n_states, 2), dtype=np.int32)
        self._pred_b = np.zeros((self.n_states, 2), dtype=np.int32)
        cnt = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            for b in range(2):
                ns = int(self._next[s, b])
                k  = cnt[ns]
                self._pred_s[ns, k] = s
                self._pred_b[ns, k] = b
                cnt[ns] += 1

    # ──────────────────────────────────────────────────────────
    #  Encoder
    # ──────────────────────────────────────────────────────────

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """
        Convolutional encode.

        Appends M = K−1 zero tail bits to drive encoder back to state 0.
        Output length = (len(info_bits) + M) × n.
        """
        bits  = np.concatenate([info_bits.astype(np.uint8),
                                 np.zeros(self.M, dtype=np.uint8)])
        state = 0
        coded = np.empty(len(bits) * self.n, dtype=np.uint8)
        idx   = 0
        for b in bits:
            out = int(self._out[state, int(b)])
            for j in range(self.n):
                coded[idx] = (out >> (self.n - 1 - j)) & 1
                idx += 1
            state = int(self._next[state, int(b)])
        return coded

    # ──────────────────────────────────────────────────────────
    #  Hard-decision Viterbi decoder
    # ──────────────────────────────────────────────────────────

    def decode(self, rx_bits: np.ndarray, n_info_bits: int) -> np.ndarray:
        """
        Hard-decision Viterbi decode (vectorised over states).

        Parameters
        ----------
        rx_bits     : 1-D uint8 array of received (noisy) coded bits.
        n_info_bits : number of information bits to recover.

        Returns
        -------
        decoded : 1-D uint8 array, length = n_info_bits.
        """
        n        = self.n
        n_states = self.n_states
        n_steps  = len(rx_bits) // n
        rx       = rx_bits[: n_steps * n].reshape(n_steps, n)

        INF = 10 ** 9
        pm  = np.full(n_states, INF, dtype=np.int64)
        pm[0] = 0

        survivor   = np.zeros((n_steps, n_states), dtype=np.uint8)
        pred_state = np.zeros((n_steps, n_states), dtype=np.int32)

        ps0 = self._pred_s[:, 0]   # shape (n_states,)
        ps1 = self._pred_s[:, 1]
        pb0 = self._pred_b[:, 0]
        pb1 = self._pred_b[:, 1]

        # Forward pass — one numpy operation per time step
        for t in range(n_steps):
            # Hamming distances for all (state, input) pairs: shape (n_states, 2)
            dists = np.sum(self._exp != rx[t], axis=2)   # broadcast over last dim

            # Accumulated metrics from both predecessors of each next-state
            m0 = pm[ps0] + dists[ps0, pb0]   # (n_states,)
            m1 = pm[ps1] + dists[ps1, pb1]   # (n_states,)

            sel   = m0 < m1                  # True → take pred-0 branch
            pm    = np.where(sel, m0, m1)
            survivor[t]   = np.where(sel, pb0, pb1)
            pred_state[t] = np.where(sel, ps0, ps1)

        # Traceback
        decoded = np.empty(n_steps, dtype=np.uint8)
        state   = int(np.argmin(pm))
        for t in range(n_steps - 1, -1, -1):
            decoded[t] = survivor[t, state]
            state       = int(pred_state[t, state])

        return decoded[:n_info_bits]


# ──────────────────────────────────────────────────────────────
#  Predefined standard codes
# ──────────────────────────────────────────────────────────────

# Rate 1/2, K=3, G = [7, 5]  (octal)
CONV_R12_K3 = ConvCode([0o7, 0o5], K=3)

# Rate 1/2, K=7, G = [133, 171]  (octal) — NASA standard (used in Voyager, 802.11a)
CONV_R12_K7 = ConvCode([0o133, 0o171], K=7)

# Rate 1/3, K=3, G = [7, 7, 5]  (octal)
CONV_R13_K3 = ConvCode([0o7, 0o7, 0o5], K=3)
