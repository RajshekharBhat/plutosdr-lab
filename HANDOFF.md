# PlutoSDR Lab — Handoff Prompt

Copy-paste this into a new Claude Code session to resume work instantly.

---

```
I am Prof. Rajshekhar V. Bhat (IIT Dharwad, ECE). I have a PlutoSDR lab project at:
  ~/Projects/plutosdr-lab/
  GitHub: https://github.com/RajshekharBhat/plutosdr-lab

Read CLAUDE.md in that directory before doing anything. Here is the context:

HARDWARE
  Two ADALM-PLUTOs connected to my Linux workstation (user-Precision-3680, Ubuntu 22.04,
  Python 3.10, NVIDIA T400 CUDA):
    TX: PlutoSDR 1 (AD9363A) — ip:192.168.2.1
    RX: PlutoSDR 2 (AD9364)  — ip:192.168.3.1
  Verify with: iio_info -s

ENVIRONMENT
  cd ~/Projects/plutosdr-lab && source venv/bin/activate
  All packages installed: pyadi-iio 0.0.20, crcmod, Pillow, commpy, torch+CUDA, numpy, scipy

PROJECT STATE (as of 2026-03-25)
  9 experiments implemented and committed to GitHub:
    01 loopback test — WORKING: BER=0/1024, EVM≈8-12%, SNR≈13-19 dB (hardware verified)
    02 text modem — full PHY: ZC preamble sync, Gardner timing, Costas phase, CRC16
    03 image over the air — JPEG packetized TX/RX
    04 OFDM — 64-FFT, cyclic prefix, pilot LS/MMSE channel estimation
    05 OFDMA — 256-FFT, 4 users, mixed modulations (BPSK/QPSK/QAM16/QAM64)
    06 channel estimation comparison — LS vs MMSE BER/NMSE curves
    07 fading demo — Rayleigh/Jake's model, multipath PSD, OTA envelope measurement
    08 ML comms — O'Shea autoencoder, learned constellation, AMC CNN classifier
    09 FEC — convolutional codes (R1/2 K3/K7, R1/3 K3) + hard-decision Viterbi, BER curves + OTA

  NEW/FIXED in this session:
    - common/fec.py: pure-NumPy ConvCode class (encoder + vectorised Viterbi decoder)
      commpy 0.1.x on PyPI is broken (Cython); fec.py is the replacement, no extra deps
    - experiments/09_fec/fec_demo.py: sim + OTA modes, BER vs Eb/N0, coding gain bar chart
    - experiments/01_loopback/loopback_test.py: FULLY FIXED with 3-stage carrier recovery:
        Stage 1: Coarse CFO from ZC preamble correlation phase slope (~−3.1 to −3.2 kHz)
        Stage 2: Fine residual CFO from BPSK squaring method (~50−130 Hz)
        Stage 3: Static phase correction — 4-quadrant search minimising Im² energy (mod 90°)
        Then: Costas loop at bw_norm=0.005 (MUST normalise amplitude to ±1 BEFORE Costas)
    - TX/RX format fixed in 6 files: complex64 input to tx(), astype(complex64) on rx() output
    - detect_preamble: fixed cosine similarity (was dividing by ||zc||² instead of ||zc||)

CARRIER RECOVERY PIPELINE (Exp 01 — the hard-won solution):
  Key insight: Costas loop error = Re(s)×Im(s) ∝ amplitude². Symbols at amplitude~18
  (before normalisation) make effective loop BW = 347× too wide → wildly unstable.
  Solution: normalise to unit amplitude BEFORE running Costas.
  Key insight 2: squaring method gives theta mod 90° — MUST try all 4 candidates and
  pick the one with minimum imaginary energy to resolve the quadrant ambiguity.
  Costas loop is at unstable equilibrium at 90° offset (error=0 there) — will NOT converge.

KNOWN NEXT STEPS
  - Build Experiment 10: live video streaming (cv2 frame → compress → packetize → TX)
  - IQ imbalance correction for QAM64+ (not yet implemented)
  - Adaptive modulation / link adaptation demo
  - GNU Radio integration (gr-iio) as optional alternative interface
  - MIMO demo using both Pluto TX+RX channels simultaneously
  - Soft-decision Viterbi (unquantized LLR input) as enhancement to Exp 09

CODING CONVENTIONS
  - All experiments import from common/ with sys.path.insert(0, "../../")
  - SPS=4 throughout, RRC α=0.35, carrier=915 MHz, fs=2 MSPS
  - TX IQ: (shaped * 2**14).astype(np.complex64), pass directly to sdr.tx(iq)
  - RX: raw = sdr.rx() returns complex128 array; do raw.astype(np.complex64)
  - Plots saved as PNG (matplotlib Agg backend, headless server)
  - Git commit after every significant change

MY PREFERENCES
  - Operate autonomously — state the plan then execute without asking at each step
  - Use sonnet-4-6 for sub-agents to conserve opus quota
  - LaTeX for any documents/reports (not markdown)
  - Parallel agents for independent subtasks

What would you like to work on?
```
