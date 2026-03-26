"""
Framing layer: preamble generation, frame building, frame detection, CRC.

Frame structure:
  [PREAMBLE 64 bits][SFD 16 bits][HEADER 16 bits][PAYLOAD N bits][CRC16 16 bits]

PREAMBLE : Zadoff-Chu sequence repeated — good autocorrelation, used for
           timing + coarse frequency offset estimation.
SFD      : Start-of-frame delimiter 0xF0F0 for bit-level sync confirmation.
HEADER   : [scheme 3b][payload_len_bytes 13b]
CRC16    : CCITT CRC over HEADER+PAYLOAD.
"""

import struct
import numpy as np
import crcmod

# ---------- CRC ----------
_crc16_fn = crcmod.predefined.mkCrcFun("crc-ccitt-false")

def crc16(data: bytes) -> int:
    return _crc16_fn(data)

def crc16_bytes(data: bytes) -> bytes:
    return struct.pack(">H", crc16(data))


# ---------- Preamble (Zadoff-Chu based) ----------
PREAMBLE_LEN   = 64        # bits after BPSK mapping → 64 chips
ZC_ROOT        = 7
SFD_BITS       = np.array([1,1,1,1, 0,0,0,0, 1,1,1,1, 0,0,0,0], dtype=np.uint8)  # 0xF0F0

SCHEME_MAP = {"BPSK": 0, "QPSK": 1, "QAM16": 2, "QAM64": 3}
SCHEME_INV = {v: k for k, v in SCHEME_MAP.items()}


def _zc_sequence(length: int, root: int = ZC_ROOT) -> np.ndarray:
    """Complex Zadoff-Chu sequence of given length.
    Odd N:  exp(-j pi r n(n+1) / N)   Even N: exp(-j pi r n^2 / N)
    """
    n = np.arange(length)
    exponent = n * (n + 1) if length % 2 == 1 else n ** 2
    return np.exp(-1j * np.pi * root * exponent / length)


def preamble_symbols() -> np.ndarray:
    """Return complex ZC preamble symbols (length = PREAMBLE_LEN)."""
    return _zc_sequence(PREAMBLE_LEN)


def build_frame(payload_bits: np.ndarray, scheme: str) -> np.ndarray:
    """
    Build a complete frame as a bit array.
    payload_bits must be a uint8 array of 0s and 1s.
    Returns bit array (uint8).
    """
    payload_bytes = np.packbits(payload_bits).tobytes()
    payload_len   = len(payload_bytes)

    scheme_id = SCHEME_MAP[scheme]
    header_val = (scheme_id << 13) | (payload_len & 0x1FFF)
    header_bytes = struct.pack(">H", header_val)

    protected = header_bytes + payload_bytes
    crc_bytes = crc16_bytes(protected)

    frame_bytes = header_bytes + payload_bytes + crc_bytes
    frame_bits  = np.unpackbits(np.frombuffer(frame_bytes, dtype=np.uint8))

    # Prepend SFD bits (preamble is sent as ZC symbols separately)
    return np.concatenate([SFD_BITS, frame_bits]).astype(np.uint8)


def parse_frame(bits: np.ndarray):
    """
    Parse a frame bit array (after SFD detection).
    Returns (payload_bits, scheme_str, crc_ok) or (None, None, False) on error.
    """
    if len(bits) < 32 + 16:   # header + min crc
        return None, None, False

    # Header
    header_val  = int(np.packbits(bits[:16]).view(">u2")[0])
    scheme_id   = (header_val >> 13) & 0x7
    payload_len = header_val & 0x1FFF
    scheme      = SCHEME_INV.get(scheme_id)
    if scheme is None:
        return None, None, False

    total_bits = 16 + payload_len * 8 + 16      # header + payload + crc
    if len(bits) < total_bits:
        return None, None, False

    frame_bytes = np.packbits(bits[:total_bits]).tobytes()
    protected   = frame_bytes[:-2]
    recv_crc    = struct.unpack(">H", frame_bytes[-2:])[0]
    calc_crc    = crc16(protected)
    crc_ok      = (recv_crc == calc_crc)

    payload_bits = bits[16 : 16 + payload_len * 8]
    return payload_bits, scheme, crc_ok


# ---------- Preamble correlation detector ----------

def detect_preamble(rx_syms: np.ndarray, threshold: float = 0.5):
    """
    Slide ZC preamble over rx_syms and return list of detected start indices
    where cosine-similarity (normalised correlation) exceeds threshold.

    Metric = |<seg, conj(zc)>| / (||seg|| * ||zc||)  ∈ [0, 1].
    1.0 = perfect match (any phase offset), 0 = uncorrelated noise.
    Typical threshold: 0.5 (loose) to 0.7 (strict).
    """
    zc        = preamble_symbols()
    L         = len(zc)
    N         = len(rx_syms)
    zc_norm   = np.linalg.norm(zc)                 # = √L  (unit-magnitude ZC)
    zc_conj   = np.conj(zc) / zc_norm              # pre-normalised template
    positions = []
    i = 0
    while i <= N - L:
        seg  = rx_syms[i : i + L]
        seg_n = np.linalg.norm(seg)
        if seg_n < 1e-9:
            i += 1
            continue
        corr = np.abs(np.dot(seg, zc_conj)) / seg_n   # cosine similarity
        if corr > threshold:
            positions.append((i, float(corr)))
            i += L       # skip past this detection to avoid duplicate hits
        else:
            i += 1
    return positions


def coarse_freq_offset(rx_preamble: np.ndarray, fs: float) -> float:
    """
    Estimate coarse carrier frequency offset from received preamble
    using the correlation phase slope method.
    Returns offset in Hz.
    """
    zc   = preamble_symbols()
    corr = rx_preamble * np.conj(zc)
    # Phase difference between first and second halves
    L    = len(corr)
    phi1 = np.angle(np.sum(corr[: L // 2]))
    phi2 = np.angle(np.sum(corr[L // 2 :]))
    dphi = np.unwrap([phi1, phi2])[1] - np.unwrap([phi1, phi2])[0]
    return dphi * fs / (2 * np.pi * (L // 2))
