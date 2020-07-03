# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt

from ..signal.signal_detrend import signal_detrend


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
    # Define window length
    if min_frequency == 0:
        min_frequency = 0.04  # sanitize lowest frequency to lf
    if window is not None:
        nperseg = int(window * sampling_rate)
    else:
        # to capture at least 5 times slowest wave-length
        nperseg = int((5 / min_frequency) * sampling_rate)

    out = {}
    return out

# =============================================================================
# Short-Time Fourier Transform (STFT)
# =============================================================================


def stft(signal, sampling_rate=1000, min_frequency=0.04, max_frequency=np.inf, overlap=None, nperseg=None, show=True):
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

# =============================================================================
# Smooth Pseudo-Wigner-Ville Distribution
# =============================================================================


def spwvd(signal, sampling_rate=1000, window_length=None, smoothing_length=None, segment_step=None, nfreqbin=None, show=True):
    """SPWVD

    References
    ----------
    J. M. O' Toole, M. Mesbah, and B. Boashash, (2008),
    "A New Discrete Analytic Signal for Reducing Aliasing in the
     Discrete Wigner-Ville Distribution", IEEE Trans.
     """

    # Define parameters
    N = len(signal)
    sample_spacing = 1 / sampling_rate
    if nfreqbin is None:
        nfreqbin = N

    # Zero-padded signal to length 2N
    signal_padded = np.append(signal, np.zeros_like(signal))

    # DFT
    signal_fft = np.fft.fft(signal_padded)
    signal_fft[1: N-1] = signal_fft[1: N-1] * 2
    signal_fft[N:] = 0

    # Inverse FFT
    signal_ifft = np.fft.ifft(signal_fft)
    signal_ifft[N:] = 0

    # Make analytic signal
    a_signal = scipy.signal.hilbert(signal_detrend(signal_ifft))

    # Create normalize windows in time and frequency
    # window
    if window_length is None:
        window_length = np.floor(N/2.)
    # Plus one if window length is odd
    if window_length % 2 == 1:
        window_length += 1
    # smoothing window
    if smoothing_length is None:
        smoothing_length = np.floor(N/5.)
    if smoothing_length % 2 == 1:
        smoothing_length += 1
    std_freq = window_length / (6 * np.sqrt(2 * np.log(2)))
    std_time = smoothing_length / (6 * np.sqrt(2 * np.log(2)))

    # Calculate windows
    w_freq = scipy.signal.gaussian(window_length, std_freq)
    w_freq /= sum(w_freq)

    w_time = scipy.signal.gaussian(smoothing_length, std_time)
    w_time /= sum(w_time)

    midpt_freq = (window_length - 1) / 2
    midpt_time = (smoothing_length - 1) / 2

    # Create arrays
    time_array = np.arange(start=0, stop=N+1, step=segment_step, dtype='int')
    frequency_array = np.fft.fftfreq(nfreqbin, sample_spacing)[0:nfreqbin / 2]
    pwvd = np.zeros(nfreqbin, len(time_array), dtype='complex')

    # Calculate pwvd
    for i, t in enumerate(time_array):
        # time shift
        tau_max = np.min(t+midpt_time-1, N-t+midpt_time, round(nfreqbin/2), midpt_frequency)
        # time-lag list
        tau = np.arange(start=-np.min(midpt_time, N-t),
                        stop=np.min(midpt_time, t-2) + 1,
                        step=1,
                        dtype='int')
        # zero frequency
        pwvd[0, i] = np.sum(2 * (
                w_time[midpt_time+tau] / np.sum(w_time[midpt_time+tau])) *
                a_signal[t-tau-1] * np.conjugate(a_signal[t-tau-1]))
        # other frequencies
        for m in range(tau_max):
            tau = np.arange(start=-np.min(midpt_time, N-t-m-1),
                            stop=np.min(midpt_time, t-m-1) + 1,
                            step=1,
                            dtype='int')

            m_time = 2 * (w_time[midpt_time+tau] / np.sum(w_time[midpt_time+tau]))

            # compute positive half
            rmm = np.sum(m_time * a_signal[t+m-tau-1] * np.conjugate(a_signal[t-m-tau]))
            pwvd[m, i] = w_freq[midpt_freq+m-1] * rmm
            # compute negative half
            rmm = np.sum(m_time * a_signal[t-m-tau] * np.conjugate(a_signal[t+m-tau-1]))
            pwvd[nfreqbin-m-1, i] = w_freq[midpt_freq-m] * rmm

        m = np.round(nfreqbin / 2)

        if t <= N-m and t >= m and m <= midpt_freq:
            tau = np.arange(start=-np.min(midpt_time, N-t-m),
                            stop=np.min(midpt_time, N-t, m) + 1,
                            step=1,
                            dtype='int')
            m_time = w_time[midpt_time+tau] / np.sum(w_time[midpt_time+tau])
            pwvd[m-1, i] = 0.5 * (
                    np.sum(w_freq[midpt_freq+m] * (m_time * a_signal[t+m-tau-1] *
                           np.conjugate(a_signal[t-m-tau]))) +
                    np.sum(w_freq[midpt_freq-m] * (m_time * a_signal[t-m-tau] *
                           np.conjugate(a_signal[t+m-tau-1]))))
    pwvd = np.fft.fft(pwvd, axis=0)
    # rotate for t=0, f=0 at lower left
    pwvd = np.rot90(pwvd.T, 1)

    return pwvd, time_array, frequency_array
