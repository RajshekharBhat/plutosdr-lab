# Experiment 07 — Wireless Channel Modelling and Fading

> **IIT Dharwad — EE/ECE SDR Teaching Lab**
> Rajshekhar V Bhat, Dept. of EECE

---

## Table of Contents

1. [The Wireless Channel Model](#1-the-wireless-channel-model)
2. [Power Delay Profile (PDP)](#2-power-delay-profile-pdp)
3. [Coherence Bandwidth](#3-coherence-bandwidth)
4. [Time-Varying Channels and the Doppler Effect](#4-time-varying-channels-and-the-doppler-effect)
5. [Jake's Scattering Model](#5-jakes-scattering-model)
6. [Coherence Time](#6-coherence-time)
7. [Rayleigh Fading](#7-rayleigh-fading)
8. [Rician Fading](#8-rician-fading)
9. [The 2x2 Fading Classification](#9-the-2x2-fading-classification)
10. [BER Under Rayleigh Fading](#10-ber-under-rayleigh-fading)
11. [OFDM and Multipath Robustness](#11-ofdm-and-multipath-robustness)
12. [Key Formulas Reference Table](#12-key-formulas-reference-table)

---

## 1. The Wireless Channel Model

A wireless channel is characterised by two independent physical phenomena:

- **Multipath propagation** — the transmitted signal arrives at the receiver via multiple paths (reflections, diffractions, scattering), each with a different time delay and attenuation. This creates *frequency selectivity* in the channel.
- **Doppler spread** — relative motion between transmitter and receiver (or scatterers) causes each multipath component to arrive with a slightly shifted carrier frequency. This creates *time variation* of the channel.

The complete baseband-equivalent channel is described by its **time-varying channel impulse response** $h(\tau, t)$, where $\tau$ is the excess delay and $t$ is the observation time. The received signal is:

$$y(t) = \int_{-\infty}^{\infty} h(\tau, t)\, x(t - \tau)\, d\tau + n(t)$$

where $x(t)$ is the transmitted baseband signal and $n(t)$ is additive white Gaussian noise.

For a discrete **tapped-delay-line** model (appropriate for sampled systems), the channel reduces to:

$$y[n] = \sum_{k=0}^{L-1} h_k[n]\, x[n - k] + \text{noise}[n]$$

where $L$ is the number of taps, $h_k[n]$ is the (possibly time-varying) complex gain of the $k$-th tap at time $n$, and each tap corresponds to a delay of $k$ samples.

The two key dimensions are:
- **Delay domain** — captures multipath structure, delay spread, frequency selectivity
- **Time domain** — captures mobility (Doppler), channel variation rate

---

## 2. Power Delay Profile (PDP)

The **Power Delay Profile** is the ensemble-average power as a function of excess delay:

$$P(\tau) = \mathbb{E}\left[\,|h(\tau, t)|^2\,\right]$$

For a tapped-delay-line model with $L$ taps at delays $\tau_0, \tau_1, \ldots, \tau_{L-1}$ and complex gains $a_0, a_1, \ldots, a_{L-1}$:

$$P_k = |a_k|^2 \quad \text{at delay } \tau_k$$

### Mean Excess Delay

The **mean excess delay** is the power-weighted mean of the tap delays:

$$\bar{\tau} = \frac{\sum_{k} P_k\, \tau_k}{\sum_{k} P_k}$$

### RMS Delay Spread

The **RMS delay spread** $\sigma_\tau$ is the square root of the second central moment of the PDP:

$$\sigma_\tau = \sqrt{\frac{\sum_{k} P_k\, \tau_k^2}{\sum_{k} P_k} - \bar{\tau}^2}$$

The RMS delay spread is the single most important parameter characterising multipath — it directly governs the onset of inter-symbol interference (ISI) and sets the coherence bandwidth.

**Our 3-tap NLOS indoor channel example:**

| Tap | Delay | Gain | Phase |
|-----|-------|------|-------|
| 0   | 0 μs  | 0 dB    | 0°  |
| 1   | 5 μs  | −6 dB   | 45° |
| 2   | 15 μs | −15 dB  | 90° |

Computed: $\bar{\tau} \approx 0.8\ \mu\text{s}$, $\sigma_\tau \approx 1.7\ \mu\text{s}$ (exact values depend on tap normalisation; `pdp_stats()` computes them precisely).

---

## 3. Coherence Bandwidth

The **coherence bandwidth** $B_c$ is the range of frequencies over which the channel transfer function is approximately constant (i.e., correlated). It is inversely proportional to the RMS delay spread.

Two standard criteria are used:

**50% correlation criterion** (commonly used for OFDM subcarrier spacing):

$$B_c \approx \frac{1}{5\, \sigma_\tau}$$

**90% correlation criterion** (stricter, used for high-modulation-order systems):

$$B_c \approx \frac{1}{2\pi\, \sigma_\tau}$$

### Flat vs Frequency-Selective Fading

The relationship between signal bandwidth $B_s$ and coherence bandwidth $B_c$ determines the type of fading:

- **Flat fading**: $B_s \ll B_c$ — all frequency components of the signal experience the same gain and phase shift. The channel can be modelled as a single complex scalar multiplier.
- **Frequency-selective fading**: $B_s \gg B_c$ — different frequency components of the signal are attenuated differently. This causes **inter-symbol interference (ISI)** in single-carrier systems.

> **Example with our channel:** $\sigma_\tau \approx 1.7\ \mu\text{s}$ gives $B_c(50\%) \approx 118\ \text{kHz}$. At 1 MSPS, our signal occupies $\sim 1\ \text{MHz}$, which is much larger than $B_c$. Hence our channel is **frequency-selective**.

---

## 4. Time-Varying Channels and the Doppler Effect

When the receiver (or scatterers) is moving with velocity $v$ relative to the transmitter, a plane wave arriving at angle $\theta$ (measured from the direction of motion) undergoes a **Doppler shift**:

$$f_{\text{Doppler}} = \frac{v}{c}\, f_c\, \cos\theta$$

where $c = 3 \times 10^8\ \text{m/s}$ is the speed of light and $f_c$ is the carrier frequency. The **maximum Doppler shift** (for $\theta = 0$, i.e., motion directly toward the source) is:

$$f_d = \frac{v\, f_c}{c}$$

### Doppler shift vs velocity at 915 MHz

| Velocity | $f_d$ |
|----------|--------|
| 5 km/h (walking)  | $\approx 4.2\ \text{Hz}$   |
| 30 km/h (cycling) | $\approx 25.4\ \text{Hz}$  |
| 100 km/h (car)    | $\approx 84.7\ \text{Hz}$  |
| 300 km/h (train)  | $\approx 254\ \text{Hz}$   |

The Doppler spread of the received signal (from all multipath components arriving at different angles) causes the channel to vary with time.

---

## 5. Jake's Scattering Model

**Jake's model** is the standard analytical model for a mobile radio channel under the assumption of **isotropic scattering** — the scatterers are uniformly distributed in angle around the mobile receiver.

The model represents the complex channel envelope as a sum of $N$ sinusoids (one per scatterer), each arriving at a uniformly distributed angle $\alpha_k = 2\pi k / N$:

$$h(t) = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} \exp\!\left(j\, 2\pi f_d \cos\!\left(\frac{2\pi k}{N} + \theta_k\right) t + j\phi_k\right)$$

where $\theta_k$ and $\phi_k$ are random initial angles (uniformly distributed in $[0, 2\pi)$), and $f_d$ is the maximum Doppler shift.

### Jake's Doppler Power Spectrum

The theoretical Doppler power spectral density resulting from isotropic scattering is the **U-shaped (Clarke) spectrum**:

$$S(f) = \frac{1}{\pi f_d \sqrt{1 - \left(\dfrac{f}{f_d}\right)^2}}, \qquad |f| < f_d$$

and $S(f) = 0$ outside $|f| < f_d$. The characteristic U-shape arises because more scatterers arrive near $\theta = 0°$ and $180°$ (perpendicular to direction of motion) than near $\theta = 90°$ (along direction of motion).

### Why Rayleigh Fading Emerges

By the **central limit theorem**, as $N \to \infty$, the sum of sinusoids converges to a **complex Gaussian** random process. The in-phase $h_I(t)$ and quadrature $h_Q(t)$ components are both zero-mean Gaussian with equal variance:

$$h(t) = h_I(t) + j\, h_Q(t), \quad h_I, h_Q \sim \mathcal{N}(0, \sigma^2)$$

The envelope $r(t) = |h(t)|$ then follows a **Rayleigh distribution** — the hallmark of fading without a line-of-sight path.

---

## 6. Coherence Time

The **coherence time** $T_c$ is the time interval over which the channel impulse response can be considered approximately constant (i.e., time-correlated). It is inversely related to the Doppler spread $f_d$.

**Clarke's result** (based on the 50% autocorrelation bandwidth of the U-shaped Doppler PSD):

$$T_c \approx \frac{0.423}{f_d}$$

### Slow vs Fast Fading

The relationship between the symbol duration $T_s$ and the coherence time $T_c$ determines how rapidly the channel varies during transmission:

- **Slow fading**: $T_s \ll T_c$ — the channel is essentially constant over many symbol periods. ISI and ICI from time variation are negligible; standard equalisers work well.
- **Fast fading**: $T_s \gg T_c$ — the channel changes significantly within a single symbol period. This causes **inter-carrier interference (ICI) in OFDM** and makes equalisation very difficult.

> **Example:** At $f_d = 5\ \text{Hz}$ (walking speed, 915 MHz), $T_c \approx 84.6\ \text{ms}$. At 1 MSPS, one symbol lasts $1\ \mu\text{s} \ll 84.6\ \text{ms}$. This is firmly **slow fading**.

---

## 7. Rayleigh Fading

**Rayleigh fading** occurs when there is **no dominant line-of-sight (LOS) path** between transmitter and receiver, and many scattered waves contribute with approximately equal amplitude.

Under Jake's model with no LOS component, the complex channel gain is:

$$h = h_I + j\, h_Q, \qquad h_I,\, h_Q \sim \mathcal{N}(0, \sigma^2)$$

### Rayleigh Distribution

The envelope $r = |h|$ follows the **Rayleigh distribution**:

$$f_R(r) = \frac{2r}{\Omega}\, \exp\!\left(-\frac{r^2}{\Omega}\right), \qquad r \geq 0$$

where $\Omega = \mathbb{E}[r^2] = 2\sigma^2$ is the **mean power** of the fading envelope.

The phase $\phi = \angle h$ is **uniformly distributed** over $[0, 2\pi)$ and is independent of $r$.

Key statistics of the Rayleigh distribution ($\Omega = 1$ for unit power):
- Mean: $\mathbb{E}[r] = \sqrt{\pi/4} \approx 0.886$
- Variance: $\text{Var}[r] = 1 - \pi/4 \approx 0.215$
- The envelope frequently takes very small values (deep fades) — this is the cause of severe BER degradation compared to AWGN.

---

## 8. Rician Fading

**Rician fading** occurs when there is a **dominant line-of-sight (LOS) component** in addition to the scattered multipath waves.

The complex channel gain is:

$$h = \underbrace{\sqrt{\frac{K}{K+1}}\, e^{j\phi_\text{LOS}}}_{\text{LOS}} + \underbrace{\frac{1}{\sqrt{K+1}}\, h_\text{scatter}}_{\text{Rayleigh scatter}}$$

where $\phi_\text{LOS}$ is the (random) LOS initial phase and $h_\text{scatter}$ is a unit-power Rayleigh-fading component.

### Rician K-factor

The **K-factor** (or Rician K-factor) is the ratio of the LOS power to the scattered power:

$$K = \frac{P_\text{LOS}}{P_\text{scatter}}$$

In decibels: $K_\text{dB} = 10\log_{10}(K)$.

- $K = 0$ ($K_\text{dB} \to -\infty$): no LOS, reduces to **Rayleigh fading**.
- $K \to \infty$: pure LOS, approaches **AWGN** (constant envelope, no fading).

### Rician Distribution

The envelope $r = |h|$ follows the **Rician distribution**:

$$f_R(r) = \frac{2r(K+1)}{\Omega}\, \exp\!\left(-K - \frac{(K+1)r^2}{\Omega}\right) I_0\!\left(2r\sqrt{\frac{K(K+1)}{\Omega}}\right), \qquad r \geq 0$$

where $\Omega = \mathbb{E}[r^2]$ is the total mean power and $I_0(\cdot)$ is the modified Bessel function of the first kind, order zero.

Typical K-factor values in practice:

| Environment | K (dB) |
|-------------|---------|
| Rich scattering, no LOS (Rayleigh) | $-\infty$ to 0 |
| Indoor, moderate LOS | 3–6 dB |
| Strong indoor LOS | 7–15 dB |
| Outdoor suburban LOS | 10–20 dB |

---

## 9. The 2x2 Fading Classification

Every wireless channel can be classified along two independent axes based on how its bandwidth and coherence time compare to the signal parameters:

|  | **Signal BW $\ll B_c$ (Flat fading)** | **Signal BW $\gg B_c$ (Frequency-selective)** |
|---|---|---|
| **$T_s \ll T_c$ (Slow fading)** | **Flat & Slow** — simplest case. Channel is a single constant complex gain per packet. No ISI, no ICI. Closest to AWGN behaviour (except for deep fades). | **Selective & Slow** — channel introduces ISI but is static over many symbols. Can be effectively equalised (ZF, MMSE) or handled with OFDM + pilot-based estimation. Most common indoor WiFi scenario. |
| **$T_s \gg T_c$ (Fast fading)** | **Flat & Fast** — no ISI, but channel varies within a symbol. In OFDM, this destroys the subcarrier orthogonality (ICI). Requires fast tracking loops. | **Selective & Fast** — both ISI and rapid channel variation. Most challenging scenario. Requires very dense pilots and advanced equalisation. Occurs in high-speed vehicular mmWave links. |

### Classification of our PlutoSDR channel (1 MSPS, $f_d = 5$ Hz):

- Signal BW $= 1\ \text{MHz} \gg B_c(50\%) \approx 118\ \text{kHz}$ → **Frequency-Selective**
- Symbol period $T_s = 1\ \mu\text{s} \ll T_c \approx 84.6\ \text{ms}$ → **Slow Fading**
- **Regime: Frequency-Selective / Slow Fading** (typical indoor NLOS)

This is the most practically important regime and is exactly the scenario where **OFDM with pilot-based channel estimation** excels.

---

## 10. BER Under Rayleigh Fading

### AWGN Reference (BPSK)

Over an ideal AWGN channel, the BER for uncoded BPSK is:

$$P_b^\text{AWGN} = \frac{1}{2}\,\text{erfc}\!\left(\sqrt{\frac{E_b}{N_0}}\right)$$

This decreases **exponentially** with $E_b/N_0$, giving a steep BER curve.

### Flat Rayleigh Fading with Perfect CSI at Receiver (BPSK)

When the channel is flat Rayleigh fading and the receiver has perfect channel state information (CSI), it can apply **maximum-ratio combining** by multiplying the received signal by $h^* / |h|$. The instantaneous BER conditioned on a channel realization $h$ is:

$$P_b(h) = \frac{1}{2}\,\text{erfc}\!\left(\sqrt{|h|^2\,\gamma_b}\right)$$

where $\gamma_b = E_b/N_0$ is the average SNR. Averaging over the Rayleigh distribution of $|h|^2$ gives the **closed-form result**:

$$P_b^\text{Rayleigh} = \frac{1}{2}\left(1 - \sqrt{\frac{\bar{\gamma}}{1 + \bar{\gamma}}}\right)$$

where $\bar{\gamma} = E_b/N_0$ (average SNR per bit). At high SNR ($\bar{\gamma} \gg 1$):

$$P_b^\text{Rayleigh} \approx \frac{1}{4\bar{\gamma}}$$

This means the Rayleigh BER curve falls as $1/\text{SNR}$ (inverse linear in dB), compared to the exponential decay $\sim e^{-\text{SNR}}$ for AWGN. This fundamental difference explains the 20–30 dB SNR gap between fading and AWGN BER curves at low BER values.

### The Diversity Order

The slope of the BER vs SNR curve on a log-log scale is called the **diversity order**. For flat Rayleigh fading with a single antenna and perfect CSI, the diversity order is **1** (one degree of freedom). Techniques such as antenna diversity, coding, or OFDM can increase the effective diversity order and close the gap to AWGN performance.

### Frequency-Selective Fading Without Equalisation

When the channel is frequency-selective and no equaliser is used, ISI creates an **irreducible BER floor**: even as SNR → ∞, the BER does not approach zero because ISI power grows along with signal power. The ISI floor level depends on the channel delay spread relative to the symbol period.

---

## 11. OFDM and Multipath Robustness

**Orthogonal Frequency Division Multiplexing (OFDM)** is the standard technique for combating frequency-selective fading in modern wireless standards (WiFi, LTE, 5G NR, DVB-T/S2).

### How OFDM Handles Multipath

The key idea is to convert a single frequency-selective wideband channel into $N$ parallel **flat-fading narrowband sub-channels**, one per subcarrier:

1. **Divide** the available bandwidth into $N$ orthogonal subcarriers, each with spacing $\Delta f = 1/(N T_s)$.
2. **Modulate** each subcarrier independently in the frequency domain. Apply the IFFT to obtain the time-domain OFDM symbol.
3. **Add a Cyclic Prefix (CP)** of length $N_\text{CP}$ samples (identical copy of the last $N_\text{CP}$ samples of the OFDM symbol prepended to the front).
4. **At the receiver**, remove the CP, apply the FFT, and each subcarrier output sees a **scalar complex gain**:

$$Y[k] = H[k]\, X[k] + W[k], \qquad k = 0, 1, \ldots, N-1$$

where $H[k] = H(k \Delta f)$ is the channel frequency response sampled at the $k$-th subcarrier.

### Cyclic Prefix Condition

For the CP to fully eliminate ISI, its length must satisfy:

$$N_\text{CP} \geq L_\text{max}$$

where $L_\text{max}$ is the length of the channel impulse response in samples.

> **For our 3-tap channel:** The maximum delay is 15 samples. With $N_\text{CP} = 16$, the condition $16 \geq 15$ is satisfied — the cyclic prefix fully contains the channel memory, eliminating ISI.

### Pilot-Based Channel Estimation in OFDM

Since each subcarrier sees a different complex gain $H[k]$, the receiver must estimate $H[k]$ for each subcarrier. This is done by inserting known **pilot symbols** at predetermined subcarrier indices:

**Least-Squares (LS) Estimation at pilots:**

$$\hat{H}_\text{LS}[k_p] = \frac{Y[k_p]}{X[k_p]}, \qquad k_p \in \mathcal{P}$$

where $\mathcal{P}$ is the set of pilot subcarrier indices and $X[k_p]$ is the known pilot value.

**Interpolation to data subcarriers:**

The LS estimates at pilot positions are interpolated (linearly or using MMSE) to obtain $\hat{H}[k]$ at all data subcarrier positions, then used for one-tap equalization:

$$\hat{X}[k] = \frac{Y[k]}{\hat{H}[k]}$$

### OFDM vs Single-Carrier: BER Comparison

| Aspect | Single-Carrier | OFDM |
|--------|----------------|------|
| Frequency-selective channel | ISI, BER floor without EQ | Converted to flat per subcarrier |
| Equaliser complexity | High (time-domain MMSE/DFE) | Low (one-tap FEQ per subcarrier) |
| PAPR | Low | High (sum of many sinusoids) |
| CP overhead | Not needed | $N_\text{CP}/N$ fractional overhead |
| Pilot overhead | Low | Depends on pilot density |

In the simulation (`fading_doppler.png`, panel [1,0]), you can directly observe that OFDM with LS channel estimation achieves near-AWGN BER on the frequency-selective 3-tap channel, while single-carrier (no equaliser) hits a BER floor above $10^{-2}$.

---

## 12. Key Formulas Reference Table

| Quantity | Symbol | Formula | Typical value (our 3-tap channel) |
|---|---|---|---|
| Mean excess delay | $\bar{\tau}$ | $\sum P_k \tau_k / \sum P_k$ | $\approx 0.8\ \mu\text{s}$ |
| RMS delay spread | $\sigma_\tau$ | $\sqrt{\sum P_k \tau_k^2 / \sum P_k - \bar{\tau}^2}$ | $\approx 1.7\ \mu\text{s}$ |
| Coherence BW (50%) | $B_c$ | $1/(5\sigma_\tau)$ | $\approx 118\ \text{kHz}$ |
| Coherence BW (90%) | $B_c$ | $1/(2\pi\sigma_\tau)$ | $\approx 94\ \text{kHz}$ |
| Maximum Doppler | $f_d$ | $v f_c / c$ | 4.2 Hz (walking, 915 MHz) |
| Coherence time | $T_c$ | $0.423 / f_d$ | 84.6 ms ($f_d = 5$ Hz), 2.1 ms ($f_d = 200$ Hz) |
| Rayleigh envelope PDF | $f_R(r)$ | $(2r/\Omega)\exp(-r^2/\Omega)$ | — |
| Rician envelope PDF | $f_R(r)$ | $(2r(K+1)/\Omega)\exp(-K-(K+1)r^2/\Omega)I_0(2r\sqrt{K(K+1)/\Omega})$ | — |
| AWGN BER (BPSK) | $P_b$ | $\frac{1}{2}\text{erfc}(\sqrt{\bar{\gamma}})$ | — |
| Rayleigh BER (BPSK, CSI) | $P_b$ | $\frac{1}{2}\left(1 - \sqrt{\frac{\bar{\gamma}}{1+\bar{\gamma}}}\right)$ | — |
| OFDM CP condition | — | $N_\text{CP} \geq L_\text{max}$ | $16 \geq 15$ (satisfied) |

---

## Further Reading

- T. S. Rappaport, *Wireless Communications: Principles and Practice*, 2nd ed., Prentice Hall, 2002. (Chapters 3, 4)
- D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*, Cambridge University Press, 2005. (Chapters 2, 3) — freely available at https://web.stanford.edu/~dntse/wireless_book.html
- A. Goldsmith, *Wireless Communications*, Cambridge University Press, 2005. (Chapters 2, 3)
- W. C. Jakes (ed.), *Microwave Mobile Communications*, IEEE Press, 1974. (The original Jake's model reference)
- R. Nee and R. Prasad, *OFDM for Wireless Multimedia Communications*, Artech House, 2000.
