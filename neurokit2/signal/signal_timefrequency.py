# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def signal_timefrequency(signal, sampling_rate=1000, min_frequency=0.04, max_frequency=np.inf, window=None, overlap=None, show=True):
    """Quantify changes of a nonstationary signal’s frequency over time.
    The objective of time-frequency analysis is to offer a more informative description of the signal
    which reveals the temporal variation of its frequency contents.

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

    Returns
    -------
    frequency : np.array
        Frequency.
    time : np.array
        Time array.
    stft : np.array
        Short Term Fourier Transform. Time increases across its columns and frequency increases
        down the rows.
    Examples
    -------
    >>> import neurokit2 as nk
    >>> import numpy as np
    >>> data = nk.data("bio_resting_5min_100hz")
    >>> sampling_rate=100
    >>> peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=sampling_rate)
    >>> peaks = np.where(peaks == 1)[0]
    >>> rri = np.diff(peaks) / sampling_rate * 1000
    >>> desired_length = int(np.rint(peaks[-1]))
    >>> signal = nk.signal_interpolate(peaks[1:], rri, x_new=np.arange(desired_length))
    >>> f, t, stft = nk.signal_timefrequency(signal, sampling_rate, max_frequency=0.5, show=True)
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

    frequency, time, stft = short_term_ft(
            signal,
            sampling_rate=sampling_rate,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            overlap=overlap,
            nperseg=nperseg,
            show=show
            )

    return frequency, time, stft

# =============================================================================
# Short-Time Fourier Transform (STFT)
# =============================================================================


def short_term_ft(signal, sampling_rate=1000, min_frequency=0.04, max_frequency=np.inf, overlap=None, nperseg=None, show=True):
    """Short-term Fourier Transform.
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


def smooth_pseudo_wvd(signal, freq_window=None, time_window=None, segment_step=1, nfreqbin=None):
    """Smoothed Pseudo Wigner Ville Distribution

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    freq_window : np.array
        Frequency smoothing window.
    time_window: np.array
        Time smoothing window
    segment_step : int
        The step between samples in `time_array`. Default to 1.
    nfreqbin : int
        Number of Frequency bins

    Returns
    -------
    frequency_array : np.array
        Frequency array.
    time_array : np.array
        Time array.
    pwvd : np.array
        SPWVD. Time increases across its columns and frequency increases
        down the rows.
    References
    ----------
    J. M. O' Toole, M. Mesbah, and B. Boashash, (2008),
    "A New Discrete Analytic Signal for Reducing Aliasing in the
     Discrete Wigner-Ville Distribution", IEEE Trans.
     """

    # Define parameters
    N = len(signal)
#    sample_spacing = 1 / sampling_rate
    if nfreqbin is None:
        nfreqbin = N

#    # Zero-padded signal to length 2N
#    signal_padded = np.append(signal, np.zeros_like(signal))
#
#    # DFT
#    signal_fft = np.fft.fft(signal_padded)
#    signal_fft[1: N-1] = signal_fft[1: N-1] * 2
#    signal_fft[N:] = 0
#
#    # Inverse FFT
#    signal_ifft = np.fft.ifft(signal_fft)
#    signal_ifft[N:] = 0
#
#    # Make analytic signal
#    a_signal = scipy.signal.hilbert(signal_detrend(signal_ifft))

    # Create normalize windows in time and frequency
    if freq_window is None:
        freq_length = np.floor(nfreqbin / 4.0)
        # Plus one if window length is odd
        if freq_length % 2 == 0:
            freq_length += 1
        freq_window = scipy.signal.hamming(int(freq_length))
    elif len(freq_window) % 2 == 0:
        raise ValueError("The length of freq_window must be odd.")

    if time_window is None:
        time_length = np.floor(N / 10.0)
        # Plus one if window length is odd
        if time_length % 2 == 0:
            time_length += 1
        time_window = scipy.signal.hamming(int(time_length))
    elif len(time_window) % 2 == 0:
        raise ValueError("The length of time_window must be odd.")

    midpt_freq = (len(freq_window) - 1) // 2
    midpt_time = (len(time_window) - 1) // 2

#    std_freq = freq_window / (6 * np.sqrt(2 * np.log(2)))
#    std_time = time_window / (6 * np.sqrt(2 * np.log(2)))
#
#    # Calculate windows
#    w_freq = scipy.signal.gaussian(freq_window, std_freq)
#    w_freq /= sum(w_freq)
#
#    w_time = scipy.signal.gaussian(time_window, std_time)
#    w_time /= sum(w_time)

    # Create arrays
    time_array = np.arange(start=0, stop=N, step=segment_step, dtype=int)
#    frequency_array = np.fft.fftfreq(nfreqbin, sample_spacing)[0:nfreqbin / 2]
    frequency_array = 0.5 * np.arange(nfreqbin, dtype=float) / nfreqbin
    pwvd = np.zeros((nfreqbin, len(time_array)), dtype=complex)

    # Calculate pwvd
    for i, t in enumerate(time_array):
        # time shift
        tau_max = np.min([t + midpt_time - 1,
                          N - t + midpt_time,
                          np.round(nfreqbin / 2.0) - 1,
                          midpt_freq])
        # time-lag list
        tau = np.arange(start=-np.min([midpt_time, N - t]),
                        stop=np.min([midpt_time, t - 1]) + 1,
                        dtype='int')
        time_pts = (midpt_time + tau).astype(int)
        g2 = time_window[time_pts]
        g2 = g2 / np.sum(g2)
        signal_pts = (t - tau - 1).astype(int)
        # zero frequency
        pwvd[0, i] = np.sum(g2 * signal[signal_pts] * np.conjugate(signal[signal_pts]))
        # other frequencies
        for m in range(int(tau_max)):
            tau = np.arange(start=-np.min(midpt_time, N - t - m),
                            stop=np.min(midpt_time, t - m - 1) + 1,
                            dtype='int')
            time_pts = (midpt_time + tau).astype(int)
            g2 = time_window[time_pts]
            g2 = g2 / np.sum(g2)
            signal_pt1 = (t + m - tau - 1).astype(int)
            signal_pt2 = (t - m - tau - 1).astype(int)
            # compute positive half
            rmm = np.sum(g2 * signal[signal_pt1] * np.conjugate(signal[signal_pt2]))
            pwvd[m + 1, i] = freq_window[midpt_freq + m + 1] * rmm
            # compute negative half
            rmm = np.sum(g2 * signal[signal_pt2] * np.conjugate(signal[signal_pt1]))
            pwvd[nfreqbin - m - 1, i] = freq_window[midpt_freq - m + 1] * rmm

        m = np.round(nfreqbin / 2.0)

        if t <= N - m and t >= m + 1 and m <= midpt_freq:
            tau = np.arange(start=-np.min([midpt_time, N - t - m]),
                            stop=np.min([midpt_time, t - 1 - m]) + 1,
                            dtype='int')
            time_pts = (midpt_time + tau + 1).astype(int)
            g2 = time_window[time_pts]
            g2 = g2 / np.sum(g2)
            signal_pt1 = (t + m - tau).astype(int)
            signal_pt2 = (t - m - tau).astype(int)
            x = np.sum(g2 * signal[signal_pt1] * np.conjugate(signal[signal_pt2]))
            x *= freq_window[midpt_freq + m + 1]
            y = np.sum(g2 * signal[signal_pt2] * np.conjugate(signal[signal_pt1]))
            y *= freq_window[midpt_freq - m + 1]
            pwvd[m, i] = 0.5 * (x + y)

    pwvd = np.real(np.fft.fft(pwvd, axis=0))

    # Visualization

    return frequency_array, time_array, pwvd
