"""
PlutoSDR hardware configuration and connection helpers.

Two-device setup:
  TX: Pluto 1  AD9363A  ip:192.168.2.1  usb:1.5.5
  RX: Pluto 2  AD9364   ip:192.168.3.1  usb:1.6.5
"""

import numpy as np
import adi

# ---------- Device URIs ----------
PLUTO_TX_URI = "ip:192.168.2.1"   # AD9363A — transmitter
PLUTO_RX_URI = "ip:192.168.3.1"   # AD9364  — receiver

# Fall-back USB URIs (use when IPs are unavailable)
PLUTO_TX_USB = "usb:1.5.5"
PLUTO_RX_USB = "usb:1.6.5"

# ---------- Defaults ----------
DEFAULT_FC      = 915e6     # 915 MHz — ISM, no licence needed indoors
DEFAULT_FS      = 2e6       # 2 MSPS sample rate
DEFAULT_BW      = 1e6       # RF bandwidth
TX_GAIN_ATT     = -30       # dB attenuation (0 = max power, -89 = min)
RX_GAIN_DB      = 50        # dB manual gain


def connect_tx(uri=PLUTO_TX_URI, fc=DEFAULT_FC, fs=DEFAULT_FS,
               bw=DEFAULT_BW, gain_att=TX_GAIN_ATT, buf_size=2**16):
    """Return a configured PlutoSDR TX (adi.Pluto) object."""
    sdr = adi.Pluto(uri=uri)
    sdr.sample_rate        = int(fs)
    sdr.tx_lo              = int(fc)
    sdr.tx_rf_bandwidth    = int(bw)
    sdr.tx_hardwaregain_chan0 = gain_att
    sdr.tx_cyclic_buffer   = False
    sdr.tx_buffer_size     = buf_size
    print(f"[TX] Connected: {uri}  fc={fc/1e6:.3f} MHz  fs={fs/1e6:.2f} MSPS  att={gain_att} dB")
    return sdr


def connect_rx(uri=PLUTO_RX_URI, fc=DEFAULT_FC, fs=DEFAULT_FS,
               bw=DEFAULT_BW, gain_db=RX_GAIN_DB, buf_size=2**16):
    """Return a configured PlutoSDR RX (adi.Pluto) object."""
    sdr = adi.Pluto(uri=uri)
    sdr.sample_rate        = int(fs)
    sdr.rx_lo              = int(fc)
    sdr.rx_rf_bandwidth    = int(bw)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0   = gain_db
    sdr.rx_buffer_size     = buf_size
    print(f"[RX] Connected: {uri}  fc={fc/1e6:.3f} MHz  fs={fs/1e6:.2f} MSPS  gain={gain_db} dB")
    return sdr


def connect_both(fc=DEFAULT_FC, fs=DEFAULT_FS, bw=DEFAULT_BW,
                 tx_uri=PLUTO_TX_URI, rx_uri=PLUTO_RX_URI,
                 gain_att=TX_GAIN_ATT, gain_db=RX_GAIN_DB, buf_size=2**16):
    """Connect both TX and RX Plutos and return (tx_sdr, rx_sdr)."""
    tx = connect_tx(tx_uri, fc, fs, bw, gain_att, buf_size)
    rx = connect_rx(rx_uri, fc, fs, bw, gain_db,  buf_size)
    return tx, rx
