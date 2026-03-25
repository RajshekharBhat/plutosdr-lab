import numpy as np


class OOK_Modulator:
    """On-Off Keying (OOK) modulation class.
    This class provides methods to modulate and demodulate binary data using OOK.
    """

    @staticmethod
    def modulate(data, amplitude=1):
        """Modulates binary data using OOK.

        Parameters:
        data (list): Binary data to be modulated (0s and 1s).
        amplitude (float): Amplitude of the signal.

        Returns:
        list: Modulated signal.
        """
        return (np.array(data) * amplitude).astype(np.complex128)

    @staticmethod
    def demodulate(signal, threshold=0.5):
        """Demodulates an OOK signal back to binary data.

        Parameters:
        signal (list): OOK modulated signal.
        threshold (float): Threshold to determine if the signal is a '1' or '0'.

        Returns:
        list: Demodulated binary data.
        """
        return [1 + 0j if sample >= threshold else 0 for sample in signal]


class BPSK_Modulator:
    """Binary Phase Shift Keying (BPSK) modulation class.
    This class provides methods to modulate and demodulate binary data using BPSK.
    """

    @staticmethod
    def modulate(bitstream):
        """Modulates binary data using BPSK.

        Parameters:
        data (list): Binary data to be modulated (0s and 1s).

        Returns:
        list: Modulated signal.
        """
        return np.array([(1 + 0j) if bit == 1 else (-1 + 0j) for bit in bitstream])

    @staticmethod
    def demodulate(signal, threshold=0):
        """Demodulates a BPSK signal back to binary data.

        Parameters:
        signal (list): BPSK modulated signal.

        Returns:
        list: Demodulated binary data.
        """
        return np.array([1 if sample.real > threshold else 0 for sample in signal])


class ASK_2_Modulator:
    """Amplitude Shift Keying (ASK) modulation class.
    """

    @staticmethod
    def modulate(bitstream, amplitudes=[-1, 1]):
        """Modulates binary data using ASK.

        Parameters:
        bitstream (list): Binary data to be modulated (0s and 1s).
        amplitudes (list): Amplitudes of the signal.

        Returns:
        list: Modulated signal.
        """
        return np.array(
            [(amplitudes[1] if bit == 1 else amplitudes[0]) for bit in bitstream],
            dtype=np.complex128,
        )

    @staticmethod
    def demodulate(signal, threshold=0):
        """Demodulates an ASK signal back to binary data.

        Parameters:
        signal (list): ASK modulated signal.
        threshold (float): Threshold to determine if the signal is a '1' or '0'.

        Returns:
        list: Demodulated binary data.
        """
        return np.array([1 if sample.real >= threshold else 0 for sample in signal])


class ASK_4_Modulator:
    """4-Level Amplitude Shift Keying (ASK) modulation class.
    """

    @staticmethod
    def modulate(bitstream, amplitudes=[-3, -1, 1, 3]):
        """Modulates binary data using 4-level ASK.

        Parameters:
        bitstream (list): Binary data to be modulated (0s and 1s).
        amplitudes (list): List of amplitudes for the four levels.

        Returns:
        list: Modulated signal.
        """
        return np.array(
            [amplitudes[int(''.join(map(str, bit)), 2)] for bit in np.reshape(bitstream, (-1, 2))],
            dtype=np.complex128,
        )

    @staticmethod
    def demodulate(signal, levels=[-3, -1, 1, 3]):
        """Demodulates a 4-level ASK signal back to binary data.

        Parameters:
        signal (list): 4-level ASK modulated signal.

        Returns:
        list: Demodulated binary data.
        """
        return np.array([format(levels.index(int(sample.real)), '02b') for sample in signal])


class QPSK_Modulator:
    """Quadrature Phase Shift Keying (QPSK) modulation class.
    This class provides methods to modulate and demodulate binary data using QPSK.
    """

    @staticmethod
    def modulate(bitstream, amplitude=1):
        """Modulates binary data using QPSK.

        Parameters:
        bitstream (list): Binary data to be modulated (0s and 1s).

        Returns:
        list: Modulated signal.
        """
        if len(bitstream) % 2 != 0:
            raise ValueError('Bitstream length must be even for QPSK modulation.')

        symbols = []
        for i in range(0, len(bitstream), 2):
            bit_pair = bitstream[i:i + 2]
            if bit_pair == [0, 0]:
                symbols.append(complex(amplitude, 0))  # 0 degrees
            elif bit_pair == [0, 1]:
                symbols.append(complex(0, amplitude))  # 90 degrees
            elif bit_pair == [1, 0]:
                symbols.append(complex(-amplitude, 0))  # 180 degrees
            elif bit_pair == [1, 1]:
                symbols.append(complex(0, -amplitude))  # 270 degrees

        return np.array(symbols, dtype=np.complex128)

    @staticmethod
    def demodulate(signal, amplitude=1):
        """Demodulates a QPSK signal back to binary data.

        Parameters:
        signal (list): QPSK modulated signal.

        Returns:
        list: Demodulated binary data.
        """
        bitstream = []
        for sample in signal:
            if sample.real > 0 and sample.imag >= 0:
                bitstream.extend([0, 0])  # 0 degrees
            elif sample.real <= 0 and sample.imag > 0:
                bitstream.extend([0, 1])  # 90 degrees
            elif sample.real < 0 and sample.imag <= 0:
                bitstream.extend([1, 0])  # 180 degrees
            elif sample.real >= 0 and sample.imag < 0:
                bitstream.extend([1, 1])  # 270 degrees

        return np.array(bitstream)


class PSK_8_Modulator:
    """8-Phase Shift Keying (PSK) modulation class.
    This class provides methods to modulate and demodulate binary data using 8-PSK.
    """

    @staticmethod
    def modulate(bitstream, amplitude=1):
        """Modulates binary data using 8-PSK.

        Parameters:
        bitstream (list): Binary data to be modulated (0s and 1s).

        Returns:
        list: Modulated signal.
        """
        if len(bitstream) % 3 != 0:
            raise ValueError('Bitstream length must be a multiple of 3 for 8-PSK modulation.')

        symbols = []
        for i in range(0, len(bitstream), 3):
            bit_triplet = bitstream[i:i + 3]
            index = int(''.join(map(str, bit_triplet)), 2)
            angle = (index * np.pi / 4)  # 45 degrees per symbol
            symbols.append(complex(amplitude * np.cos(angle), amplitude * np.sin(angle)))

        return np.array(symbols, dtype=np.complex128)

    @staticmethod
    def demodulate(signal, amplitude=1):
        """Demodulates an 8-PSK signal back to binary data.

        Parameters:
        signal (list): 8-PSK modulated signal.

        Returns:
        list: Demodulated binary data.
        """
        bitstream = []
        for sample in signal:
            angle = np.angle(sample) % (2 * np.pi)
            index = int(round(angle / (np.pi / 4))) % 8
            bitstream.extend(format(index, '03b'))  # Convert index to 3-bit binary

        return np.array(bitstream)
