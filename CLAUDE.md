# PlutoSDR Lab — CLAUDE.md

## Identity / Owner
- **Prof. Rajshekhar V. Bhat**, Dept. of EECE, IIT Dharwad
- GitHub: `RajshekharBhat`  |  Repo: `plutosdr-lab`

---

## Hardware

| Role | Model | IP | USB | Notes |
|------|-------|----|-----|-------|
| TX (Pluto 1) | AD9363A | `ip:192.168.2.1` | `usb:1.5.5` | serial: `104473a69c0e...` |
| RX (Pluto 2) | AD9364  | `ip:192.168.3.1` | `usb:1.6.5` | serial: `1044732a9811...` |

Host: `user-Precision-3680` — Ubuntu 22.04, Python 3.10, NVIDIA T400 (CUDA 12.x)

Verify both devices are present:
```bash
iio_info -s    # should list 4 contexts (2 USB + 2 IP)
```

---

## Environment

```bash
cd ~/Projects/plutosdr-lab
source venv/bin/activate          # Python 3.10 venv
# Key packages: pyadi-iio 0.0.20, crcmod, Pillow, opencv-headless, commpy, numpy, scipy, torch+CUDA
```

Recreate venv if needed:
```bash
python3 -m venv venv --system-site-packages
pip install -r requirements.txt
```

---

## Project Architecture

```
plutosdr-lab/
├── common/                     ← shared DSP/comms library (import from any experiment)
│   ├── pluto_config.py         — connect_tx(), connect_rx(), connect_both(); default RF params
│   ├── modulation.py           — BPSK/QPSK/QAM16/QAM64 (Gray-coded); RRC pulse shaping
│   ├── framing.py              — ZC preamble (len=64), SFD, frame build/parse, CRC16-CCITT
│   ├── dsp.py                  — AGC, DC block, Gardner TED, Costas loop, FLL, M2M4 SNR est.
│   ├── channel.py              — AWGN, multipath FIR, Rayleigh (Jake's), LS/MMSE estimation
│   └── fec.py                  — ConvCode class: rate-1/n encoder + hard-decision Viterbi; CONV_R12_K3/K7, CONV_R13_K3
│
└── experiments/
    ├── 01_loopback/            — hardware sanity: EVM, SNR, constellation, PSD
    ├── 02_text_modem/          — full PHY TX+RX: text over the air, all recovery loops
    ├── 03_image_video/         — JPEG image over the air (packetized, partial recovery)
    ├── 04_ofdm/                — 64-FFT OFDM, cyclic prefix, pilot channel estimation
    ├── 05_ofdma/               — 256-FFT OFDMA, 4 users, mixed modulations
    ├── 06_channel_estimation/  — LS vs MMSE: BER and NMSE curves vs SNR
    ├── 07_fading_demo/         — Rayleigh/Jake's model, BER under fading, OTA envelope
    ├── 08_ml_comms/            — O'Shea autoencoder, learned constellation, AMC/CNN
    └── 09_fec/                 — convolutional FEC: BER vs Eb/N0, coding gain, OTA test
```

---

## Default RF Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Carrier frequency | 915 MHz | ISM band — no indoor licence needed |
| Sample rate | 2 MSPS | change with `--fs` flag |
| TX attenuation | −30 dB | 0 = max power, −89 = min; pass as `gain_att` |
| RX gain | 50 dB | manual mode; pass as `gain_db` |
| Samples per symbol | 4 | SPS=4 throughout all experiments |
| RRC roll-off | α = 0.35 | matches pulse shaper and matched filter |

---

## Frame / PHY Stack

```
[ZC Preamble — 64 complex symbols]   ← timing sync + coarse CFO
[SFD — 16 bits: 0xF0F0]              ← frame boundary confirmation
[Header — 16 bits: scheme(3b) + payload_len_bytes(13b)]
[Payload — N bits]
[CRC16-CCITT — 16 bits over Header+Payload]
```

RX chain (in order):
1. DC removal → AGC
2. Coarse CFO from ZC preamble correlation
3. FLL (second-order) for residual CFO
4. RRC matched filter
5. Gardner TED timing recovery (SPS=4)
6. Costas loop phase recovery (order 2=BPSK, 4=QPSK)
7. ZC preamble detection (normalised correlation threshold=0.5)
8. SFD scan → parse_frame() → CRC check

---

## Key Code Pointers

| What you need | Where |
|---------------|-------|
| Connect both Plutos | `common/pluto_config.py` → `connect_both()` |
| Build a TX frame | `common/framing.py` → `build_frame(bits, scheme)` |
| Modulate bits | `common/modulation.py` → `bits_to_symbols()` + `upsample_and_shape()` |
| Demodulate | `common/modulation.py` → `match_filter()` + `symbols_to_bits()` |
| Timing recovery | `common/dsp.py` → `gardner_timing_recovery()` |
| Phase recovery | `common/dsp.py` → `CostasLoop` |
| Channel models | `common/channel.py` → `awgn()`, `rayleigh_fading()`, `multipath_channel()` |
| OFDM frame | `experiments/04_ofdm/ofdm_modem.py` → `bits_to_ofdm_frame()` |
| FEC encode/decode | `common/fec.py` → `ConvCode.encode()` / `.decode()` |
| Predefined codes | `common/fec.py` → `CONV_R12_K3`, `CONV_R12_K7`, `CONV_R13_K3` |

---

## Quick-start Commands

```bash
# 1. Verify both Plutos are alive
venv/bin/python experiments/01_loopback/loopback_test.py

# 2. Text modem — open two terminals
venv/bin/python experiments/02_text_modem/transmit.py "Hello IIT Dharwad!" --cyclic --scheme QPSK
venv/bin/python experiments/02_text_modem/receive.py --scheme QPSK

# 3. Image over the air
venv/bin/python experiments/03_image_video/image_tx.py photo.jpg --quality 30
venv/bin/python experiments/03_image_video/image_rx.py --out received.jpg

# 4. OFDM simulation (no hardware needed)
venv/bin/python experiments/04_ofdm/ofdm_modem.py --mode sim --snr 20

# 5. OFDMA multi-user demo
venv/bin/python experiments/05_ofdma/ofdma_demo.py --snr 20

# 6. Channel estimation curves
venv/bin/python experiments/06_channel_estimation/channel_estimation_demo.py

# 7. Fading demo (simulation + OTA)
venv/bin/python experiments/07_fading_demo/fading_demo.py --mode sim
venv/bin/python experiments/07_fading_demo/fading_demo.py --mode ota --fc 915e6

# 8. ML autoencoder
venv/bin/python experiments/08_ml_comms/autoencoder_comms.py --mode train --epochs 50
venv/bin/python experiments/08_ml_comms/autoencoder_comms.py --mode amc

# 9. FEC (convolutional codes + Viterbi)
venv/bin/python experiments/09_fec/fec_demo.py --mode sim
venv/bin/python experiments/09_fec/fec_demo.py --mode ota --fc 915e6
```

---

## Known Issues / Next Steps

- **commpy 0.1.x (PyPI) is broken** — Cython compilation issue; `common/fec.py` (pure NumPy) replaces it for Exp 09
- **Exp 03 video** (not yet built): use `cv2.VideoCapture` + frame-by-frame encoding on top of image_tx pattern
- **OTA timing offset** between the two Plutos can cause preamble detection to miss — tuning `threshold` in `detect_preamble()` is the first dial to turn
- **IQ imbalance correction** not yet implemented — relevant at high modulation orders (64QAM+)
- **Exp 10 ideas**: MIMO (both Plutos as full-duplex via self-interference cancellation), LDPC FEC, adaptive modulation (link adaptation), GNU Radio integration, live video streaming
