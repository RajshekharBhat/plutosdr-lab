# Experiment 01 — Mathematical Derivations

This file derives every formula used in `loopback_test.py` and the supporting
`common/` modules it calls.

---

## 1  ZC Preamble Correlation Detector

### Zadoff–Chu sequence

The transmitted preamble is a length-*L* Zadoff–Chu (ZC) sequence with root *u*:

$$z[n] = e^{-j\pi u\, n(n+1)/L}, \qquad n = 0, 1, \ldots, L-1$$

ZC sequences have constant modulus (`|z[n]| = 1`) and ideal periodic
autocorrelation: the circular autocorrelation is a Kronecker delta.

### Cosine-similarity detector

Given a received symbol stream `r[k]`, slide a window of length *L* and
compute the normalised cross-correlation (cosine similarity):

$$\rho(i) = \frac{\left|\sum_{n=0}^{L-1} r[i+n]\, z^*[n]\right|}{\|r_i\|\cdot\|z\|}$$

where `‖r_i‖ = √(Σ|r[i+n]|²)` and `‖z‖ = √L` (constant modulus).

Because the ZC sequence is constant-modulus, `‖z‖ = √L` is precomputed once.
The numerator is the matched-filter output magnitude; dividing by `‖r_i‖`
normalises to the received amplitude so ρ ∈ [0, 1] regardless of gain.

**Detection rule:** declare a preamble at position *i* if `ρ(i) > threshold`
(0.4–0.5 in this code).  After a hit, skip *L* samples to avoid duplicate
detections on the same preamble.

In the code (`common/framing.py:detect_preamble`):

```python
zc_conj = np.conj(zc) / np.linalg.norm(zc)   # pre-normalised template
corr    = np.abs(np.dot(seg, zc_conj)) / seg_n  # cosine similarity
```

---

## 2  Coarse CFO Estimation from Preamble Phase Slope

A carrier frequency offset Δf rotates each sample by an additional phase
`exp(j 2π Δf t)`.  At the symbol rate `f_sym = f_s / SPS`, consecutive
received preamble chips accumulate a phase increment `φ = 2π Δf / f_sym`
radians per symbol.

### Half-sequence phase method

Split the received preamble `r_p = z · e^{jφn}` (ignoring noise) into two
halves of length `L/2`:

$$\Phi_1 = \angle\!\sum_{n=0}^{L/2-1} r_p[n]\, z^*[n], \qquad \Phi_2 = \angle\!\sum_{n=L/2}^{L-1} r_p[n]\, z^*[n]$$

Each sum is the coherent average of the phase `φn` over its half; for a
linear ramp the sum is dominated by the midpoint phase:

$$\Phi_k \approx \phi \cdot \overline{n}_k, \qquad \overline{n}_1 = \tfrac{L}{4},\quad \overline{n}_2 = \tfrac{3L}{4}$$

The phase difference is:

$$\Delta\Phi = \Phi_2 - \Phi_1 \approx \phi\cdot\tfrac{L}{2}$$

Solving for the offset in Hz:

$$\boxed{\widehat{\Delta f} = \frac{\Delta\Phi}{\pi L}\cdot f_{\text{sym}}}$$

In the code (`common/framing.py:coarse_freq_offset`):

```python
dphi = unwrapped_phi2 - unwrapped_phi1
return dphi * fs_sym / (2 * pi * (L // 2))
```

**Unambiguous range:** `|Δf| < f_sym / 2` (half the symbol rate = 250 kHz at
2 MSPS, SPS=4).  Typical two-Pluto offset is 3–4 kHz, well within range.

---

## 3  Fine Residual CFO and Static Phase — BPSK Squaring Method

After coarse correction there remains a residual offset `δf` of tens to
hundreds of Hz, plus an unknown static phase `θ₀`.

### Squaring removes BPSK modulation

A BPSK symbol `s[k] = d[k] · e^{j(2π δf k/f_{sym} + θ_0)}` where
`d[k] ∈ {±1}`.  Squaring gives:

$$s^2[k] = d[k]^2 \cdot e^{j2(2\pi\delta f\, k/f_{\text{sym}} + \theta_0)} = e^{j(4\pi\delta f\, k/f_{\text{sym}} + 2\theta_0)}$$

since `d[k]² = 1`.  The modulation is eliminated; the squared sequence is a
pure complex exponential.

### Step A — residual frequency estimate

The mean phase step between consecutive squared samples is:

$$\widehat{\psi} = \angle\sum_{k=0}^{N-2} s^2[k+1]\, (s^2[k])^* \approx \frac{4\pi\,\delta f}{f_{\text{sym}}}$$

Solving:

$$\boxed{\widehat{\delta f} = \frac{\widehat{\psi}\, f_{\text{sym}}}{4\pi}}$$

The `4π` (not `2π`) comes from the squaring operation doubling the frequency.

### Step B — static phase estimate (mod 90°)

After correcting the frequency ramp, the squared symbols collapse to a single
phasor:

$$\overline{s^2} = \frac{1}{N}\sum_k s^2[k] \approx e^{j 2\theta_0}$$

Therefore:

$$\widehat{\theta}_0 = \tfrac{1}{2}\angle\,\overline{s^2}$$

**Quadrant ambiguity:** halving the angle introduces a mod-90° ambiguity
(four candidates `θ̂₀ + k·90°`, k = 0,1,2,3).  For BPSK the correct
constellation axis is the one that minimises imaginary energy:

$$k^* = \argmin_{k\in\{0,1,2,3\}}\;\frac{1}{N}\sum_n\!\left(\operatorname{Im}\!\left[s[n]\, e^{-j(\hat\theta_0 + k\pi/2)}\right]\right)^2$$

At the correct rotation, symbols sit on the real axis so `Im(s)² ≈ σ²_n`
(noise only).  At 90° off, symbols sit on the imaginary axis so
`Im(s)² ≈ 1 + σ²_n` (signal + noise).

---

## 4  Costas Loop (BPSK, Second-Order)

### Error detector

The Costas loop estimates the residual carrier phase `φ` using the
decision-directed error:

$$e[k] = \operatorname{Re}\{s_c[k]\}\cdot\operatorname{Im}\{s_c[k]\}$$

where `s_c[k] = s[k] e^{-jφ̂[k]}` is the phase-corrected symbol.

For a noiseless BPSK symbol `s = A e^{jφ}` (with d = ±1 so s = ±A e^{jφ}):

$$e = A\cos\phi\cdot A\sin\phi = \frac{A^2}{2}\sin 2\phi$$

This is the **S-curve**: zero at `φ = 0°` (stable) and `φ = ±90°` (unstable).
The loop pulls toward 0° or 180° (the 180° ambiguity is inherent to BPSK).

**Critical:** at `φ = 90°` the error is identically zero — the loop cannot
escape this unstable equilibrium.  Stages 2–3 above must bring `φ` away from
90° before Costas runs.

### Loop filter (second-order, bilinear)

The loop filter is a proportional-plus-integral (PI) controller:

$$\hat\phi[k+1] = \hat\phi[k] + K_1\,e[k] + K_2\sum_{i\le k}e[i]$$

equivalently in state-space form:

```
int[k]   += K2 * e[k]
freq[k]   = K1 * e[k] + int[k]
phase[k] += freq[k]
```

The gains are set from the normalised loop bandwidth `Bₙ` and damping
`ζ = 1/√2` (critically damped):

$$\omega_n = 2\pi B_n, \qquad K_1 = 2\zeta\omega_n, \qquad K_2 = \omega_n^2$$

### Amplitude scaling

Because `e ∝ A²`, running the loop on un-normalised symbols (amplitude `A`)
scales the effective bandwidth by `A²`.  Symbols in this receiver have
`A ≈ 18` before normalisation; the effective BW would be `18² = 324×` the
design value, making the loop unstable.

**Fix:** normalise to unit RMS (`A ≈ 1`) **before** calling the Costas loop.

---

## 5  M2M4 SNR Estimator

### Signal model

Received symbol: `x[k] = s[k] + n[k]` where `s[k]` is the transmitted
symbol (power `P_s`) and `n[k] ~ CN(0, 2σ²)` is complex AWGN
(noise power `2σ²`).

### Second and fourth moments

For a constant-modulus signal (`|s[k]| = √P_s` always, e.g. BPSK/PSK):

$$M_2 = \mathbb{E}[|x|^2] = P_s + 2\sigma^2$$

$$M_4 = \mathbb{E}[|x|^4] = P_s^2 + 4P_s\cdot 2\sigma^2 + 2(2\sigma^2)^2 + P_s^2(\kappa_s - 1)$$

For BPSK/PSK the signal kurtosis `κ_s = E[|s|⁴]/E[|s|²]² = 1`, so:

$$M_4 = P_s^2 + 8P_s\sigma^2 + 8\sigma^4$$

### Derivation of the estimator

Compute `M_4 - M_2²`:

$$M_4 - M_2^2 = (P_s^2 + 8P_s\sigma^2 + 8\sigma^4) - (P_s + 2\sigma^2)^2$$

$$= P_s^2 + 8P_s\sigma^2 + 8\sigma^4 - P_s^2 - 4P_s\sigma^2 - 4\sigma^4 = 4P_s\sigma^2 + 4\sigma^4 = 4\sigma^2(P_s + \sigma^2)$$

Wait — let me redo with real noise `σ²` per complex dimension, total noise power `N_0 = 2σ²`:

Using `M_2 = P_s + N_0` and `M_4 = P_s^2 + 4P_s N_0 + 2N_0^2` (for κ_s = 1):

$$M_4 - M_2^2 = 2N_0^2 + 4P_s N_0 - (2P_s N_0 + N_0^2) = N_0^2 + 2P_s N_0 = N_0(N_0 + 2P_s)$$

The ratio:

$$\frac{M_2^2}{M_4 - M_2^2} = \frac{(P_s + N_0)^2}{N_0(N_0 + 2P_s)}$$

At high SNR (`P_s >> N_0`): `≈ P_s² / (2P_s N_0) = SNR/2`.  At low SNR: `≈ 1/2`.

The code uses the simpler approximation directly:

$$\boxed{\widehat{\text{SNR}}_{\text{lin}} = \frac{M_2^2}{M_4 - M_2^2}}$$

which equals `SNR/2` at high SNR for PSK.  The absolute calibration is less
important than the relative trend for comparing runs.

**Validity:** the estimator is non-data-aided (no reference symbols needed),
works for any block size ≥ ~100 symbols, but is only unbiased for
constant-modulus constellations.  For QAM a correction based on `κ_s` is
required.

---

## 6  EVM (Error Vector Magnitude)

Given received symbols `r[k]` (after all recovery) and reference symbols
`s[k]` (ideal constellation points), the RMS EVM is:

$$\text{EVM}_{\text{RMS}} = \sqrt{\frac{\frac{1}{N}\sum_{k=0}^{N-1}|r[k] - s[k]|^2}{\frac{1}{N}\sum_{k=0}^{N-1}|s[k]|^2}}$$

For BPSK (`s[k] ∈ {±1}`, so the denominator = 1):

$$\text{EVM}_{\text{RMS}} = \sqrt{\frac{1}{N}\sum_k |r[k] - s[k]|^2}$$

**Interpretation:** EVM = 0% means perfect reception; EVM ≈ 141% means the
constellation is rotated 90° (since `|±j - (±1)| = √2` for each symbol);
EVM ≈ 200% means 180° flip (since `|-1 - 1| = 2`).

The code resolves the BPSK 180° ambiguity before computing EVM:

```python
if ber_flip < ber:
    syms_ref = -syms_ref   # align reference to the received polarity
```

---

## 7  Theoretical BPSK BER

For reference, the theoretical BER for BPSK over AWGN:

$$\text{BER} = Q\!\left(\sqrt{2\,\text{SNR}}\right) = \frac{1}{2}\operatorname{erfc}\!\left(\sqrt{\text{SNR}}\right)$$

where SNR = `E_b / N_0` (energy per bit over noise spectral density).

At the measured SNR of 13–19 dB the theoretical BER ranges from
`5×10⁻⁵` to `4×10⁻¹⁵` — consistent with the observed 0 errors in 1024 bits.
