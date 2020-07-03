# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt


def signal_timefrequency(signal, sampling_rate=1000, min_frequency=0.04, max_frequency=np.inf, window=None, overlap=None, show=True):
    """Quantify changes of a nonstationary signal’s frequency over time.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    min_frequency : float
        The minimum frequency.
    max_frequency : float
        The maximum frequency.
    window : int
        Length of each segment in seconds. If None (default), window will be automatically
        calculated.
    overlap : int
        Number of points to overlap between segments. If None, noverlap = nperseg // 8. Defaults to None.
        When specified, the Constant OverLap Add (COLA) constraint must be met.
    show : bool
        If True, will return two PSD plots.

    """
    # Initialize empty container for results
    out = {}
    return out

# =============================================================================
# Short-Time Fourier Transform (STFT)
# =============================================================================

def stft(signal, sampling_rate=1000, window=None, min_frequency=0.04, max_frequency=np.inf, overlap=None, show=True):
    """Short-term
    Examples
    -------
    >>> import neurokit2 as nk
    >>> data = nk.data("bio_resting_5min_100hz")
    >>> sampling_rate=1000
    >>> sampling_rate=100
    >>> peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=sampling_rate)
    >>> peaks = np.where(peaks == 1)[0]
    >>> peaks = np.where(peaks == 1)[0]
    >>> rri = np.diff(peaks) / sampling_rate * 1000
    >>> desired_length = int(np.rint(peaks[-1] / sampling_rate * sampling_rate))
    >>> signal = nk.signal_interpolate(peaks[1:], rri, x_new=np.arange(desired_length))
    >>> f, t, stft = stft(signal, sampling_rate, max_frequency=10)
    """

    # Define window length
    if min_frequency == 0:
        min_frequency = 0.04  # sanitize lowest frequency to lf
    if window is not None:
        nperseg = int(window * sampling_rate)
    else:
        # to capture at least 5 times slowest wave-length
        nperseg = int((5 / min_frequency) * sampling_rate)

    # Check COLA
    if overlap is not None:
        if not scipy.signal.check_COLA(scipy.signal.hann(nperseg, sym=True), nperseg, overlap):
            raise ValueError("The Constant OverLap Add (COLA) constraint is not met")

    frequency, time, stft = scipy.signal.spectrogram(
        signal,
        fs=sampling_rate,
        window='hann',
        scaling='density',
        nperseg=nperseg,
        nfft=None,
        detrend=False,
        noverlap=overlap
    )

    # Visualization

    if show is True:
        lower_bound = len(frequency) - len(frequency[frequency > min_frequency])
        f = frequency[(frequency > min_frequency) & (frequency < max_frequency)]
        z = stft[lower_bound:lower_bound + len(f)]

        fig = plt.figure()
        spec = plt.pcolormesh(time, f, np.abs(z),
                              cmap=plt.get_cmap("magma"))
        plt.colorbar(spec)
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (sec)')

        fig, ax = plt.subplots()
        for i in range(len(time)):
            ax.plot(f, np.abs(z[:, i]), label="Segment" + str(np.arange(len(time))[i] + 1))
        ax.legend()
        ax.set_title('Power Spectrum Density (PSD)')
        ax.set_ylabel('PSD (ms^2/Hz)')
        ax.set_xlabel('Frequency (Hz)')

    return frequency, time, stft
