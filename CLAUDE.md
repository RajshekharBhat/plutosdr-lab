# PlutoSDR Lab — CLAUDE.md

## Hardware
- **TX**: PlutoSDR 1 (AD9363A) — `ip:192.168.2.1` / `usb:1.5.5`
- **RX**: PlutoSDR 2 (AD9364)  — `ip:192.168.3.1` / `usb:1.6.5`
- Host: Linux workstation `user-Precision-3680`, Python 3.10, CUDA (NVIDIA T400)

## Environment
```bash
source venv/bin/activate   # or use venv/bin/python directly
```

## Project Structure
```
common/           — shared DSP/comms library
  pluto_config.py — connect_tx(), connect_rx(), connect_both()
  modulation.py   — BPSK/QPSK/QAM16/QAM64, RRC pulse shaping
  framing.py      — ZC preamble, frame build/parse, CRC16, SFD detection
  dsp.py          — AGC, Gardner timing, Costas loop, FLL, SNR estimation
  channel.py      — AWGN, multipath, Rayleigh fading, LS/MMSE estimation

experiments/
  01_loopback/        — hardware verification, EVM, SNR
  02_text_modem/      — full PHY stack TX+RX, text over the air
  03_image_video/     — JPEG image over the air (packetized)
  04_ofdm/            — OFDM with pilot-based channel estimation
  05_ofdma/           — multi-user OFDMA (4 users, different modulations)
  06_channel_estimation/ — LS vs MMSE BER/NMSE comparison
  07_fading_demo/     — Rayleigh/multipath sim + OTA envelope measurement
  08_ml_comms/        — autoencoder end-to-end learning + AMC classifier
```

## Default RF parameters
- Carrier frequency: **915 MHz** (ISM, no licence needed indoors)
- Sample rate: **2 MSPS**
- TX attenuation: **-30 dB** (increase magnitude = less attenuation → more power)
- RX gain: **50 dB** manual

## Quick-start commands
```bash
# Verify hardware
venv/bin/python experiments/01_loopback/loopback_test.py

# Text modem (two terminals)
venv/bin/python experiments/02_text_modem/transmit.py "Hello!" --cyclic
venv/bin/python experiments/02_text_modem/receive.py

# OFDM simulation
venv/bin/python experiments/04_ofdm/ofdm_modem.py --mode sim --snr 20

# Fading demo
venv/bin/python experiments/07_fading_demo/fading_demo.py

# ML autoencoder
venv/bin/python experiments/08_ml_comms/autoencoder_comms.py --mode train --epochs 50
```

## Key implementation notes
- All `connect_*` helpers in `common/pluto_config.py`
- Frame = [ZC preamble (complex, 64 syms)] + [SFD bits] + [Header] + [Payload] + [CRC16]
- Timing recovery: Gardner TED → use SPS=4 throughout
- IQ from PlutoSDR: `rx()[0]` = I channel, `rx()[1]` = Q channel (int16)
- TX: scale to int16 range (×2^14), pass as `[iq, iq]` for single-channel
