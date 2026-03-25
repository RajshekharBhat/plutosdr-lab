# Experiment 10 — Mathematical Derivations

## 1  Throughput Budget

### Raw symbol rate

With sample rate `f_s = 2 MSPS` and `SPS = 4` samples per symbol:

$$R_{\text{sym}} = \frac{f_s}{\text{SPS}} = \frac{2 \times 10^6}{4} = 500\,\text{ksym/s}$$

Bits per symbol: BPSK = 1, QPSK = 2.  Raw bit rates:

| Scheme | Bits/sym | Raw bit rate |
|--------|----------|--------------|
| BPSK   | 1        | 500 kbps     |
| QPSK   | 2        | 1 Mbps       |

### PHY frame overhead

Each packet of `P` payload bytes goes through the framing layer:

| Field             | Size                        |
|-------------------|-----------------------------|
| ZC preamble       | 64 symbols (transmitted separately) |
| SFD               | 16 bits                     |
| Header            | 16 bits                     |
| Payload           | `8P` bits                   |
| CRC-16            | 16 bits                     |
| **Total bits**    | `8P + 48`                   |
| **Total symbols** | `64 + (8P + 48) / bps`      |

For BPSK with `P = 256` byte payload:

$$N_{\text{sym}} = 64 + (8 \times 256 + 48) / 1 = 64 + 2096 = 2160 \text{ sym}$$

Transmission time per packet:

$$T_{\text{pkt}} = N_{\text{sym}} / R_{\text{sym}} = 2160 / 500\,000 = 4.32 \text{ ms}$$

Payload efficiency (fraction of bits that carry user data):

$$\eta = \frac{8P}{8P + 48 + 64 \cdot \text{bps}} = \frac{2048}{2048 + 48 + 64} \approx 95.6\%$$

### Effective payload rate

$$R_{\text{payload}} = \eta \cdot R_{\text{raw}} = 0.956 \times 500 = 478 \text{ kbps (BPSK)}$$

### Frame rate estimate

A 320×240 JPEG at quality 20 typically occupies 4–8 KB.  Taking 6 KB as a
representative value:

$$N_{\text{pkt}} = \lceil 6144 / 256 \rceil = 24 \text{ packets/frame}$$

$$T_{\text{frame}} = N_{\text{pkt}} \times T_{\text{pkt}} = 24 \times 4.32 = 104 \text{ ms/frame}$$

$$\text{Frame rate} \approx 1 / 0.104 \approx \mathbf{9.6\ fps}$$

For QPSK (2× the bit rate) the same frame takes ~52 ms, giving ~19 fps —
but QPSK requires higher SNR; BPSK is recommended for robust OTA operation.

---

## 2  JPEG Compression Rate-Distortion Trade-off

The JPEG quality factor `q ∈ [1, 95]` scales the quantisation step `Q`:

$$Q(q) \approx Q_{\text{base}} / q \quad (\text{heuristic, ITU T.81 tables})$$

Higher `q` → smaller `Q` → finer quantisation → larger file.  For a
typical indoor scene at 320×240, empirical JPEG size scales roughly as:

$$\text{bytes} \approx k \cdot q^{1.5}$$

where `k ≈ 15`–25 for natural images.  The relationship is content-dependent;
textureless scenes compress much better.

**Practical trade-off for this lab:**

| Quality | Typical size | Packets (256 B) | BPSK fps |
|---------|-------------|-----------------|----------|
| 10      | ~2 KB       | 8               | ~28 fps  |
| 20      | ~5 KB       | 20              | ~11 fps  |
| 40      | ~12 KB      | 48              | ~5 fps   |
| 70      | ~30 KB      | 118             | ~2 fps   |

Choose `quality=15`–`20` for smooth video; `quality=40`+ for
inspection-quality stills.

---

## 3  Packet Loss and Partial Frame Recovery

Each packet carries an independent CRC-16.  A frame is recoverable only
if **all** its packets are received without CRC errors.

Let `p_e` = probability of packet loss (CRC failure).  For a frame with
`N_p` packets the frame success probability is:

$$P_{\text{frame}} = (1 - p_e)^{N_p}$$

| `p_e`  | `N_p = 20` | `N_p = 48` |
|--------|------------|------------|
| 0.01   | 81.8%      | 61.8%      |
| 0.001  | 98.0%      | 95.3%      |
| 0.0001 | 99.8%      | 99.5%      |

This motivates keeping `N_p` small (lower JPEG quality or smaller `PKT_SIZE`).
Forward error correction (Exp 09) would allow partial recovery.

---

## 4  Nyquist Rate and Minimum Required Bandwidth

The root-raised-cosine pulse shaping with roll-off `α = 0.35` and symbol
rate `R_sym = 500 ksym/s` occupies a two-sided bandwidth of:

$$B = R_{\text{sym}}(1 + \alpha) = 500\,000 \times 1.35 = 675 \text{ kHz}$$

The ISM band at 915 MHz has a 26 MHz allocation — the signal fits
comfortably and the Pluto's 56 MHz instantaneous bandwidth is not a
constraint.

The **minimum (Nyquist) bandwidth** for distortion-free transmission is:

$$B_{\min} = R_{\text{sym}} / 2 = 250 \text{ kHz}$$

The roll-off `α` trades excess bandwidth (35% here) for reduced ISI
sensitivity to timing errors.

---

## 5  Video Packet Header Format

```
Byte offset  Field           Size  Notes
──────────────────────────────────────────────
0–1          MARKER          2 B   0xC0 0xFE
2–3          frame_id        2 B   big-endian uint16, wraps at 65535
4–5          n_pkts          2 B   total packets in this frame
6–7          pkt_idx         2 B   index of this packet (0-based)
8–9          payload_len     2 B   bytes of JPEG data in this packet
10–…         JPEG payload    ≤ 256 B
```

Total header overhead per packet: 10 bytes out of 266 bytes = 3.8%.
