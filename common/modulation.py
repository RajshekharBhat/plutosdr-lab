"""
Modulation / demodulation: BPSK, QPSK, 16-QAM, 64-QAM.
All constellations are normalised to unit average power.
"""

import numpy as np

# ---------- Bit-to-symbol maps (Gray-coded) ----------

def _qam_constellation(order):
    """Gray-coded square QAM constellation, normalised to unit power."""
    m = int(np.sqrt(order))
    assert m * m == order and (int(np.log2(order)) % 2 == 0)
    pts = np.arange(m) - (m - 1) / 2.0
    I, Q = np.meshgrid(pts, -pts)
    # Gray code per axis
    def gray(n): return n ^ (n >> 1)
    gi = np.array([gray(i) for i in range(m)])
    gq = np.array([gray(i) for i in range(m)])
    bits_i = int(np.log2(m))
    table = {}
    for qi in range(m):
        for ii in range(m):
            idx = (gq[qi] << bits_i) | gi[ii]
            table[idx] = complex(I[qi, ii], Q[qi, ii])
    arr = np.array([table[k] for k in sorted(table)])
    arr /= np.sqrt(np.mean(np.abs(arr) ** 2))
    return arr


BPSK_TABLE  = np.array([-1.0 + 0j, 1.0 + 0j])
QPSK_TABLE  = np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=complex) / np.sqrt(2)
QAM16_TABLE = _qam_constellation(16)
QAM64_TABLE = _qam_constellation(64)

CONSTELLATIONS = {
    "BPSK":  (BPSK_TABLE,  1),
    "QPSK":  (QPSK_TABLE,  2),
    "QAM16": (QAM16_TABLE, 4),
    "QAM64": (QAM64_TABLE, 6),
}


# ---------- Core helpers ----------

def bits_to_symbols(bits: np.ndarray, scheme: str) -> np.ndarray:
    """Map bits → complex symbols."""
    table, bps = CONSTELLATIONS[scheme]
    n = len(bits)
    pad = (-n) % bps
    bits = np.concatenate([bits, np.zeros(pad, dtype=int)])
    indices = bits.reshape(-1, bps).dot(1 << np.arange(bps - 1, -1, -1))
    return table[indices]


def symbols_to_bits(syms: np.ndarray, scheme: str) -> np.ndarray:
    """Hard-decision demodulation: complex symbols → bits."""
    table, bps = CONSTELLATIONS[scheme]
    # Nearest-neighbour decision
    dist = np.abs(syms[:, None] - table[None, :])
    idx  = np.argmin(dist, axis=1)
    bits = ((idx[:, None] >> np.arange(bps - 1, -1, -1)) & 1).reshape(-1)
    return bits.astype(np.uint8)


def bits_per_symbol(scheme: str) -> int:
    return CONSTELLATIONS[scheme][1]


# ---------- Pulse shaping ----------

def rrc_filter(sps: int, alpha: float = 0.35, num_taps: int = 128) -> np.ndarray:
    """Root-raised-cosine FIR filter coefficients."""
    t = np.arange(-num_taps // 2, num_taps // 2 + 1) / sps
    with np.errstate(divide="ignore", invalid="ignore"):
        numer = np.sin(np.pi * t * (1 - alpha)) + 4 * alpha * t * np.cos(np.pi * t * (1 + alpha))
        denom = np.pi * t * (1 - (4 * alpha * t) ** 2)
        h = np.where(np.abs(denom) < 1e-8,
                     1.0 - alpha + 4 * alpha / np.pi,
                     numer / denom)
    # t = ±1/(4α) singularity
    t_sing = 1.0 / (4 * alpha) if alpha > 0 else 1e18
    mask = np.abs(np.abs(t) - t_sing) < 1e-6
    h[mask] = (alpha / np.sqrt(2)) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
        (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
    )
    h /= np.sqrt(np.sum(h ** 2))
    return h


def upsample_and_shape(symbols: np.ndarray, sps: int,
                       alpha: float = 0.35, num_taps: int = 128) -> np.ndarray:
    """Upsample by sps and apply RRC pulse shaping."""
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    h = rrc_filter(sps, alpha, num_taps)
    return np.convolve(up, h, mode="full")


def match_filter(rx: np.ndarray, sps: int,
                 alpha: float = 0.35, num_taps: int = 128) -> np.ndarray:
    """Apply matched (RRC) filter."""
    h = rrc_filter(sps, alpha, num_taps)
    return np.convolve(rx, h, mode="full")
