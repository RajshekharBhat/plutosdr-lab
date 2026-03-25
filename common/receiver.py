"""
Common single-buffer demodulator.

demodulate_buffer(raw, fs, scheme, sps) → list of (payload_bits, scheme_str)

Uses the 3-stage carrier recovery pipeline validated in Exp 01:
  1. Coarse CFO from ZC preamble phase-slope method
  2. Fine residual CFO + static phase via BPSK/QPSK Nth-power method
  3. Costas loop (order matched to scheme) — AFTER amplitude normalisation

Call this from any experiment that receives a single sdr.rx() buffer.
"""

import numpy as np
from .modulation import match_filter, rrc_filter, symbols_to_bits
from .framing    import (preamble_symbols, detect_preamble,
                          coarse_freq_offset,
                          parse_frame, SFD_BITS, PREAMBLE_LEN)
from .dsp        import (remove_dc, agc, correct_freq_offset, CostasLoop)

SPS = 4


def _find_best_preamble(mf, threshold=0.4):
    """Try all SPS timing phases; return (syms, preamble_pos, corr, phase)."""
    zc   = preamble_symbols()
    best = (0.0, 0, 0, None)
    for phase in range(SPS):
        syms = mf[phase::SPS]
        for pos, corr in detect_preamble(syms, threshold=threshold):
            if corr > best[0]:
                best = (corr, phase, pos, syms)
    corr, phase, pos, syms = best
    return (syms, pos, corr, phase) if syms is not None else (None, 0, 0.0, 0)


def _nth_power_correction(syms, order, n_est=256):
    """
    Nth-power method (order=2 for BPSK, order=4 for QPSK).
    Returns (fo_fine_hz, best_theta_rad).
    """
    fs_sym = 2e6 / SPS
    n  = min(n_est, len(syms))
    sp = syms[:n] ** order                        # removes modulation

    # Step A: fine frequency from consecutive phase steps
    phi_diff = np.angle(np.sum(sp[1:] * np.conj(sp[:-1])))
    fo_fine  = phi_diff * fs_sym / (2 * np.pi * order)

    # Apply frequency ramp at symbol level
    t = np.arange(len(syms)) / fs_sym
    syms_fc = syms * np.exp(-1j * 2 * np.pi * fo_fine * t)

    # Step B: static phase (mod 360°/order) — try all candidates
    sp2         = syms_fc[:n] ** order
    theta_base  = np.angle(np.mean(sp2)) / order
    step        = np.pi / (order // 2)            # 90° for BPSK, 45° for QPSK
    best_score  = np.inf
    best_theta  = theta_base
    for k in range(order):
        cand    = theta_base + k * step
        rotated = syms_fc[:n] * np.exp(-1j * cand)
        # Minimise imaginary energy (symbols should sit on real/imag axes)
        score   = np.mean(np.imag(rotated) ** 2)
        if score < best_score:
            best_score = score
            best_theta = cand

    return fo_fine, best_theta, syms_fc


def demodulate_buffer(raw: np.ndarray, fs: float, scheme: str,
                      sps: int = SPS,
                      threshold: float = 0.4) -> list:
    """
    Full receive chain for one sdr.rx() buffer.

    Parameters
    ----------
    raw      : complex array from sdr.rx()  (dtype complex128 or complex64)
    fs       : sample rate in Hz
    scheme   : 'BPSK', 'QPSK', 'QAM16', 'QAM64'
    sps      : samples per symbol (must match TX, default 4)
    threshold: preamble detection cosine-similarity threshold

    Returns
    -------
    List of (payload_bits, scheme_str) tuples for every valid CRC frame found.
    """
    order = 2 if scheme == "BPSK" else 4   # Nth-power order
    costas_order = 2 if scheme == "BPSK" else 4

    x     = raw.astype(np.complex64)
    x     = remove_dc(x)
    x     = agc(x)

    delay = len(rrc_filter(sps)) - 1

    # ---- Pass 1: find best timing phase + coarse CFO ----
    mf1   = match_filter(x, sps)[delay:]
    syms1, pre_pos1, corr1, phase1 = _find_best_preamble(mf1, threshold)
    if syms1 is None:
        return []

    # Coarse CFO from preamble segment (symbol-rate data, pass fs/sps)
    pre_seg_raw = mf1[phase1::sps][pre_pos1 : pre_pos1 + PREAMBLE_LEN]
    if len(pre_seg_raw) < PREAMBLE_LEN:
        return []
    fo_coarse = coarse_freq_offset(pre_seg_raw, fs / sps)

    # ---- Pass 2: re-detect after coarse CFO correction ----
    x2    = correct_freq_offset(x, fo_coarse, fs)
    mf2   = match_filter(x2, sps)[delay:]
    syms2, pre_pos2, corr2, phase2 = _find_best_preamble(mf2, threshold)
    if syms2 is None:
        return []

    zc_len   = PREAMBLE_LEN
    data_syms = syms2[pre_pos2 + zc_len :]

    if len(data_syms) < 32:
        return []

    # ---- Pass 3: fine CFO + static phase (Nth-power method) ----
    fo_fine, theta_stat, data_fc = _nth_power_correction(data_syms, order)
    data_corr = data_fc * np.exp(-1j * theta_stat)

    # ---- Normalise amplitude, then Costas ----
    rms        = np.sqrt(np.mean(np.abs(data_corr) ** 2))
    data_corr  = data_corr / (rms + 1e-9)
    costas     = CostasLoop(order=costas_order, bw_norm=0.005)
    data_corr  = costas.step(data_corr)

    # ---- Scan for SFD + parse frames ----
    bits    = symbols_to_bits(data_corr, scheme)
    sfd     = SFD_BITS
    results = []
    j = 0
    while j < len(bits) - len(sfd) - 32:
        if np.array_equal(bits[j : j + len(sfd)], sfd):
            payload_bits, sch, crc_ok = parse_frame(bits[j + len(sfd):])
            if crc_ok and payload_bits is not None:
                results.append((payload_bits, sch))
                j += len(sfd) + 16 + len(payload_bits) + 16
            else:
                j += 1
        else:
            j += 1

    return results
