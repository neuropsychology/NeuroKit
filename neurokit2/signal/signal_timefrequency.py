# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal


def signal_timefrequency(signal, sampling_rate=1000, min_frequency=0, max_frequency=np.inf, window=None):
    """Quantify changes of a nonstationary signal’s frequency over time.
    """
    # Initialize empty container for results
    out = {}
    return out

# =============================================================================
# Short-Time Fourier Transform (STFT)
# =============================================================================

def stft(signal, sampling_rate=1000, window=None):

    # Define window length
    if min_frequency == 0:
        min_frequency = 0.001  # sanitize lowest frequency
    if window is not None:
        nperseg = int(window * sampling_rate)
    else:
        # to capture at least 5 times slowest wave-length
        nperseg = int((5 / min_frequency) * sampling_rate)

    frequency, time, stft = scipy.signal.stft(
        signal,
        fs=sampling_rate,
        window='hann',
        nperseg=nperseg,
        nfft=None,
        detrend=False,
        average="mean",
        padded=True
    )

    return frequency, time, stft
