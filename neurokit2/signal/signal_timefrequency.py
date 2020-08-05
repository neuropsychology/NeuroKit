# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from ..signal.signal_detrend import signal_detrend


def signal_timefrequency(signal, sampling_rate=1000, min_frequency=0.04, max_frequency=None, method="stft", window=None, window_type='hann', mode='psd', nfreqbin=None, overlap=None, analytical_signal=True, show=True):
    """Quantify changes of a nonstationary signal’s frequency over time.
    The objective of time-frequency analysis is to offer a more informative description of the signal
    which reveals the temporal variation of its frequency contents.

    There are many different Time-Frequency Representations (TFRs) available:

    - Linear TFRs: efficient but create tradeoff between time and frequency resolution
        - Short Time Fourier Transform (STFT): the time-domain signal is windowed into short segments
        and FT is applied to each segment, mapping the signal into the TF plane. This method assumes
        that the signal is quasi-stationary (stationary over the duration of the window). The width
        of the window is the trade-off between good time (requires short duration window) versus good
        frequency resolution (requires long duration windows)
        - Wavelet Transform (WT): similar to STFT but instead of a fixed duration window functrion, a
        varying window length by scaling the axis of the window is used. At low frequency, WT proves
        high spectral resolution but poor temporal resolution. On the other hand, for high frequencies,
        the WT provides high temporal resolution but poor spectral resolution.

    - Quadratic TFRs: better resolution but computationally expensive and suffers from having
    cross terms between multiple signal components
        - Wigner Ville Distribution (WVD): while providing very good resolution in time and frequency
        of the underlying signal structure, because of its bilinear nature, existence of negative values,
        the WVD has misleading TF results in the case of multi-component signals such as EEG due to the
        presence of cross terms and inference terms. Cross WVD terms can be reduced by using moothing kernal
        functions as well as analyzing the analytic signal (instead of the original signal)
        - Smoothed Pseudo Wigner Ville Distribution (SPWVD): to address the problem of cross-terms
        suppression, SPWVD allows two independent analysis windows, one in time and the other in frequency
        domains.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    method : str
        Time-Frequency decomposition method.
    min_frequency : float
        The minimum frequency.
    max_frequency : float
        The maximum frequency.
    window : int
        Length of each segment in seconds. If None (default), window will be automatically
        calculated. For stft method
    window_type : str
        Type of window to create, defaults to 'hann'. See ``scipy.signal.get_window()`` to see full
        options of windows. For stft method.
    mode : str
        Type of return values for stft method. Can be 'psd', 'complex' (default, equivalent to output of
        stft with no padding or boundary extension), 'magnitude', 'angle', 'phase'. Default to 'psd'.
    nfreqbin : int, float
        Number of frequency bins. If None (default), nfreqbin will be set to 0.5*sampling_rate.
    overlap : int
        Number of points to overlap between segments. If None, noverlap = nperseg // 8. Defaults to None.
    analytical_signal : bool
        If True, analytical signal instead of actual signal is used in Wigner Ville Distrubution
        methods.
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
    >>> signal = nk.signal_simulate(100, sampling_rate=100, frequency=10.0)
    >>> signal += 2 * nk.signal_simulate(100, sampling_rate=100, frequency=3.0)
    >>> sampling_rate=100
    >>> f, t, stft = nk.signal_timefrequency(signal, sampling_rate, max_frequency=20, method="stft", show=True)
    >>> f, t, cwtm = nk.signal_timefrequency(signal, sampling_rate, max_frequency=20, method="cwt", show=True)
    >>> f, t, wvd = nk.signal_timefrequency(signal, sampling_rate, max_frequency=20, method="wvd", show=True)
    >>> f, t, pwvd = nk.signal_timefrequency(signal, sampling_rate, max_frequency=20, method="pwvd", show=True)
    """
    # Initialize empty container for results
    # Define window length
    if min_frequency == 0:
        min_frequency = 0.04  # sanitize lowest frequency to lf
    if max_frequency is None:
        max_frequency = sampling_rate // 2  # nyquist

    # STFT
    if method.lower() in ["stft"]:

        frequency, time, tfr = short_term_ft(
                signal,
                sampling_rate=sampling_rate,
                overlap=overlap,
                window=window,
                mode=mode,
                min_frequency=min_frequency,
                window_type=window_type
                )
    # CWT
    elif method.lower() in ["cwt", "wavelet"]:
        frequency, time, tfr = continuous_wt(
                signal,
                sampling_rate=sampling_rate,
                min_frequency=min_frequency,
                max_frequency=max_frequency
                )
    # WVD
    elif method in ["WignerVille", "wvd"]:
        frequency, time, tfr = wvd(
                signal,
                sampling_rate=sampling_rate,
                n_freqbins=nfreqbin,
                analytical_signal=analytical_signal,
                method="WignerVille"
                )
    # pseudoWVD
    elif method in ["pseudoWignerVille", "pwvd"]:
        frequency, time, tfr = wvd(
                signal,
                sampling_rate=sampling_rate,
                n_freqbins=nfreqbin,
                analytical_signal=analytical_signal,
                method="pseudoWignerVille"
                )

    # Sanitize output
    lower_bound = len(frequency) - len(frequency[frequency >= min_frequency])
    f = frequency[(frequency >= min_frequency) & (frequency <= max_frequency)]
    z = tfr[lower_bound:lower_bound + len(f)]


    if show is True:
        plot_timefrequency(
                z,
                time,
                f,
                signal=signal,
                method=method,
                )


    return f, time, z

# =============================================================================
# Short-Time Fourier Transform (STFT)
# =============================================================================


def short_term_ft(signal, sampling_rate=1000, min_frequency=0.04, overlap=None,
                  window=None, window_type='hann', mode='psd'):
    """Short-term Fourier Transform.
    """

    if window is not None:
        nperseg = int(window * sampling_rate)
    else:
        # to capture at least 5 times slowest wave-length
        nperseg = int((2 / min_frequency) * sampling_rate)

    frequency, time, tfr = scipy.signal.spectrogram(
        signal,
        fs=sampling_rate,
        window=window_type,
        scaling='density',
        nperseg=nperseg,
        nfft=None,
        detrend=False,
        noverlap=overlap,
        mode=mode
    )

    return frequency, time, np.abs(tfr)


# =============================================================================
# Continuous Wavelet Transform (CWT) - Morlet
# =============================================================================


def continuous_wt(signal, sampling_rate=1000, min_frequency=0.04, max_frequency=None, nfreqbin=None):
    """Continuous Wavelet Transform.

    References
    ----------
    - Neto, O. P., Pinheiro, A. O., Pereira Jr, V. L., Pereira, R., Baltatu, O. C., & Campos, L. A. (2016).
    Morlet wavelet transforms of heart rate variability for autonomic nervous system activity.
    Applied and Computational Harmonic Analysis, 40(1), 200-206.

   - Wachowiak, M. P., Wachowiak-Smolíková, R., Johnson, M. J., Hay, D. C., Power, K. E.,
   & Williams-Bell, F. M. (2018). Quantitative feature analysis of continuous analytic wavelet transforms
   of electrocardiography and electromyography. Philosophical Transactions of the Royal Society A:
   Mathematical, Physical and Engineering Sciences, 376(2126), 20170250.
    """

    # central frequency
    w = 6.  # recommended

    if nfreqbin is None:
        nfreqbin = sampling_rate // 2

    # frequency
    frequency = np.linspace(min_frequency, max_frequency, nfreqbin)

    # time
    time = np.arange(len(signal)) / sampling_rate
    widths = w * sampling_rate / (2 * frequency * np.pi)

    # Mother wavelet = Morlet
    tfr = scipy.signal.cwt(signal, scipy.signal.morlet2, widths, w=w)

    return frequency, time, np.abs(tfr)


# =============================================================================
# Wigner-Ville Distribution
# =============================================================================
def wvd(signal, sampling_rate=1000, n_freqbins=None, analytical_signal=True, method="WignerVille"):
    """Wigner Ville Distribution and Pseudo-Wigner Ville Distribution.
    """
    # Compute the analytical signal
    if analytical_signal:
        signal = scipy.signal.hilbert(signal_detrend(signal))

    # Pre-processing
    if n_freqbins is None:
        n_freqbins = 256

    if method in ["pseudoWignerVille", "pwvd"]:
        fwindows = np.zeros(n_freqbins + 1)
        fwindows_mpts = len(fwindows) // 2
        windows_length = n_freqbins // 4
        windows_length = windows_length - windows_length % 2 + 1
        windows = scipy.hamming(windows_length)
        fwindows[fwindows_mpts + np.arange(- windows_length // 2, windows_length // 2)] = windows
    else:
        fwindows = np.ones(n_freqbins + 1)
        fwindows_mpts = len(fwindows) // 2

    time = np.arange(len(signal)) * 1.0 / sampling_rate

    # This is discrete frequency (should we return?)
    if n_freqbins % 2 == 0:
        frequency = np.hstack((np.arange(n_freqbins / 2),
                               np.arange(-n_freqbins / 2, 0)))
    else:
        frequency = np.hstack((np.arange((n_freqbins - 1) / 2),
                               np.arange(-(n_freqbins - 1) / 2, 0)))
    tfr = np.zeros((n_freqbins, time.shape[0]), dtype=complex)  # the time-frequency matrix

    tausec = round(n_freqbins / 2.0)
    winlength = tausec - 1
    # taulens: len of tau for each step
    taulens = np.min(np.c_[np.arange(signal.shape[0]),
                           signal.shape[0] - np.arange(signal.shape[0]) - 1,
                           winlength * np.ones(time.shape)], axis=1)
    conj_signal = np.conj(signal)
    # iterate and compute the wv for each indices
    for idx in range(time.shape[0]):
        tau = np.arange(-taulens[idx], taulens[idx] + 1).astype(int)
        # this step is required to use the efficient DFT
        indices = np.remainder(n_freqbins + tau, n_freqbins).astype(int)
        tfr[indices, idx] = fwindows[fwindows_mpts + tau] * signal[idx + tau] * conj_signal[idx - tau]
        if (idx < signal.shape[0] - tausec) and (idx >= tausec + 1):
            tfr[tausec, idx] = fwindows[fwindows_mpts + tausec] * signal[idx + tausec] * \
                np.conj(signal[idx - tausec]) + \
                fwindows[fwindows_mpts - tausec] * signal[idx - tausec] * conj_signal[idx + tausec]
            tfr[tausec, idx] *= 0.5

    # Now tfr contains the product of the signal segments and its conjugate.
    # To find wd we need to apply fft one more time.
    tfr = np.fft.fft(tfr, axis=0)
    tfr = np.real(tfr)

    # continuous time frequency
    frequency = 0.5 * np.arange(n_freqbins, dtype=float) / n_freqbins * sampling_rate

    return frequency, time, tfr


# =============================================================================
# Smooth Pseudo-Wigner-Ville Distribution
# =============================================================================


def smooth_pseudo_wvd(signal, sampling_rate=1000, freq_length=None, time_length=None, segment_step=1, nfreqbin=None, window_method="hamming"):
    """Smoothed Pseudo Wigner Ville Distribution

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    freq_length : np.ndarray
        Lenght of frequency smoothing window.
    time_length: np.array
        Lenght of time smoothing window
    segment_step : int
        The step between samples in `time_array`. Default to 1.
    nfreqbin : int
        Number of Frequency bins.
    window_method : str
        Method used to create smoothing windows. Can be "hanning"/ "hamming" or "gaussian".

    Returns
    -------
    frequency_array : np.ndarray
        Frequency array.
    time_array : np.ndarray
        Time array.
    pwvd : np.ndarray
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
    # sample_spacing = 1 / sampling_rate
    if nfreqbin is None:
        nfreqbin = 300

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
    signal = scipy.signal.hilbert(signal_detrend(signal_ifft))

    # Create smoothing windows in time and frequency
    if freq_length is None:
        freq_length = np.floor(N / 4.0)
        # Plus one if window length is not odd
        if freq_length % 2 == 0:
            freq_length += 1
    elif len(freq_length) % 2 == 0:
        raise ValueError("The length of frequency smoothing window must be odd.")

    if time_length is None:
        time_length = np.floor(N / 10.0)
        # Plus one if window length is not odd
        if time_length % 2 == 0:
            time_length += 1
    elif len(time_length) % 2 == 0:
        raise ValueError("The length of time smoothing window must be odd.")

    if window_method == "hamming":
        freq_window = scipy.signal.hamming(int(freq_length))  # normalize by max
        time_window = scipy.signal.hamming(int(time_length))  # normalize by max
    elif window_method == "gaussian":
        std_freq = freq_length / (6 * np.sqrt(2 * np.log(2)))
        freq_window = scipy.signal.gaussian(freq_length, std_freq)
        freq_window /= max(freq_window)
        std_time = time_length / (6 * np.sqrt(2 * np.log(2)))
        time_window = scipy.signal.gaussian(time_length, std_time)
        time_window /= max(time_window)
    # to add warning if method is not one of the supported methods

    # Mid-point index of windows
    midpt_freq = (len(freq_window) - 1) // 2
    midpt_time = (len(time_window) - 1) // 2

    # Create arrays
    time_array = np.arange(start=0, stop=N, step=segment_step, dtype=int) / sampling_rate
    # frequency_array = np.fft.fftfreq(nfreqbin, sample_spacing)[0:nfreqbin / 2]
    frequency_array = 0.5 * np.arange(nfreqbin, dtype=float) / N
    pwvd = np.zeros((nfreqbin, len(time_array)), dtype=complex)

    # Calculate pwvd
    for i, t in enumerate(time_array):
        # time shift
        tau_max = np.min([t + midpt_time - 1,
                          N - t + midpt_time,
                          np.round(N / 2.0) - 1,
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
            tau = np.arange(start=-np.min([midpt_time, N - t - m]),
                            stop=np.min([midpt_time, t - m - 1]) + 1,
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

        m = np.round(N / 2.0)

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


# =============================================================================
# Plot function
# =============================================================================
def plot_timefrequency(z, time, f, signal=None, method="stft"):
    """Visualize a time-frequency matrix.
    """

    if method == "stft":
        figure_title = "Short-time Fourier Transform Magnitude"
        fig, ax = plt.subplots()
        for i in range(len(time)):
            ax.plot(f, z[:, i], label="Segment" + str(np.arange(len(time))[i] + 1))
        ax.legend()
        ax.set_title('Signal Spectrogram')
        ax.set_ylabel('STFT Magnitude')
        ax.set_xlabel('Frequency (Hz)')

    elif method == "cwt":
        figure_title = "Continuous Wavelet Transform Magnitude"
    elif method == "wvd":
        figure_title = "Wigner Ville Distrubution Spectrogram"
        fig = plt.figure()
        plt.plot(time, signal)
        plt.xlabel('Time (sec)')
        plt.ylabel('Signal')

    elif method == "pwvd":
        figure_title = "Pseudo Wigner Ville Distribution Spectrogram"

    fig, ax = plt.subplots()
    spec = ax.pcolormesh(time, f, z, cmap=plt.get_cmap("magma"))
    plt.colorbar(spec)
    ax.set_title(figure_title)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (sec)')
    return fig
