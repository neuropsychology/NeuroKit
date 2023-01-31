# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from ..signal import signal_smooth


def ppg_findpeaks(ppg_cleaned, sampling_rate=1000, method="elgendi", show=False, **kwargs):
    """**Find systolic peaks in a photoplethysmogram (PPG) signal**

    Parameters
    ----------
    ppg_cleaned : Union[list, np.array, pd.Series]
        The cleaned PPG channel as returned by :func:`.ppg_clean`.
    sampling_rate : int
        The sampling frequency of the PPG (in Hz, i.e., samples/second). The default is 1000.
    method : str
        The processing pipeline to apply. Can be one of ``"elgendi"``, ``"bishop"``. The default is
        ``"elgendi"``.
    show : bool
        If ``True``, returns a plot of the thresholds used during peak detection. Useful for
        debugging. The default is ``False``.

    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the samples at which systolic
        peaks occur, accessible with the key ``"PPG_Peaks"``.

    See Also
    --------
    ppg_simulate, ppg_clean

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk
      import matplotlib.pyplot as plt

      ppg = nk.ppg_simulate(heart_rate=75, duration=20, sampling_rate=50)
      ppg_clean = nk.ppg_clean(ppg, sampling_rate=50)

      @savefig p_ppg_findpeaks1.png scale=100%
      peaks = nk.ppg_findpeaks(ppg_clean, sampling_rate=100, show=True)
      @suppress
      plt.close()

      # Method by Bishop et al., (2018)
      @savefig p_ppg_findpeaks2.png scale=100%
      peaks = nk.ppg_findpeaks(ppg, method="bishop", show=True)
      @suppress
      plt.close()


    References
    ----------
    * Elgendi, M., Norton, I., Brearley, M., Abbott, D., & Schuurmans, D. (2013). Systolic peak
      detection in acceleration photoplethysmograms measured from emergency responders in tropical
      conditions. PloS one, 8(10), e76585.
    * Bishop, S. M., & Ercole, A. (2018). Multi-scale peak and trough detection optimised for
      periodic and quasi-periodic neuroscience data. In Intracranial Pressure & Neuromonitoring XVI
      (pp. 189-195). Springer International Publishing.

    """
    method = method.lower()
    if method in ["elgendi"]:
        peaks = _ppg_findpeaks_elgendi(ppg_cleaned, sampling_rate, show=show, **kwargs)
    elif method in ["msptd", "bishop2018", "bishop"]:
        peaks, _ = _ppg_findpeaks_bishop(ppg_cleaned, show=show, **kwargs)
    else:
        raise ValueError("`method` not found. Must be one of the following: 'elgendi', 'bishop'.")

    # Prepare output.
    info = {"PPG_Peaks": peaks}

    return info


def _ppg_findpeaks_elgendi(
    signal,
    sampling_rate=1000,
    peakwindow=0.111,
    beatwindow=0.667,
    beatoffset=0.02,
    mindelay=0.3,
    show=False,
):
    """Implementation of Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection in
    Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical Conditions. PLoS ONE 8(10): e76585.
    doi:10.1371/journal.pone.0076585.

    All tune-able parameters are specified as keyword arguments. `signal` must be the bandpass-filtered raw PPG
    with a lowcut of .5 Hz, a highcut of 8 Hz.

    """
    if show:
        _, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax0.plot(signal, label="filtered")

    # Ignore the samples with negative amplitudes and square the samples with
    # values larger than zero.
    signal_abs = signal.copy()
    signal_abs[signal_abs < 0] = 0
    sqrd = signal_abs**2

    # Compute the thresholds for peak detection. Call with show=True in order
    # to visualize thresholds.
    ma_peak_kernel = int(np.rint(peakwindow * sampling_rate))
    ma_peak = signal_smooth(sqrd, kernel="boxcar", size=ma_peak_kernel)

    ma_beat_kernel = int(np.rint(beatwindow * sampling_rate))
    ma_beat = signal_smooth(sqrd, kernel="boxcar", size=ma_beat_kernel)

    thr1 = ma_beat + beatoffset * np.mean(sqrd)  # threshold 1

    if show:
        ax1.plot(sqrd, label="squared")
        ax1.plot(thr1, label="threshold")
        ax1.legend(loc="upper right")

    # Identify start and end of PPG waves.
    waves = ma_peak > thr1
    beg_waves = np.where(np.logical_and(np.logical_not(waves[0:-1]), waves[1:]))[0]
    end_waves = np.where(np.logical_and(waves[0:-1], np.logical_not(waves[1:])))[0]
    # Throw out wave-ends that precede first wave-start.
    end_waves = end_waves[end_waves > beg_waves[0]]

    # Identify systolic peaks within waves (ignore waves that are too short).
    num_waves = min(beg_waves.size, end_waves.size)
    min_len = int(np.rint(peakwindow * sampling_rate))  # this is threshold 2 in the paper
    min_delay = int(np.rint(mindelay * sampling_rate))
    peaks = [0]

    for i in range(num_waves):

        beg = beg_waves[i]
        end = end_waves[i]
        len_wave = end - beg

        if len_wave < min_len:
            continue

        # Visualize wave span.
        if show:
            ax1.axvspan(beg, end, facecolor="m", alpha=0.5)

        # Find local maxima and their prominence within wave span.
        data = signal[beg:end]
        locmax, props = scipy.signal.find_peaks(data, prominence=(None, None))

        if locmax.size > 0:
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            # Enforce minimum delay between peaks.
            if peak - peaks[-1] > min_delay:
                peaks.append(peak)

    peaks.pop(0)

    if show:
        ax0.scatter(peaks, signal_abs[peaks], c="r")
        ax0.legend(loc="upper right")
        ax0.set_title("PPG Peaks (Method by Elgendi et al., 2013)")

    peaks = np.asarray(peaks).astype(int)
    return peaks


def _ppg_findpeaks_bishop(
    signal,
    show=False,
):
    """Implementation of Bishop SM, Ercole A (2018) Multi-scale peak and trough detection optimised
    for periodic and quasi-periodic neuroscience data. doi:10.1007/978-3-319-65798-1_39.

    Currently designed for short signals of relatively low sampling frequencies (e.g. 6 seconds at
    100 Hz). Also, the function currently only returns peaks, but it does identify pulse onsets too.
    """

    # TODO: create ppg_peaks() that also returns onsets and stuff

    # Setup
    N = len(signal)
    L = int(np.ceil(N / 2) - 1)

    # Step 1: calculate local maxima and local minima scalograms

    # - detrend: this removes the best-fit straight line
    x = scipy.signal.detrend(signal, type="linear")

    # - initialise LMS matrices
    m_max = np.full((L, N), False)
    m_min = np.full((L, N), False)

    # - populate LMS matrices
    for k in range(1, L):  # scalogram scales
        for i in range(k + 2, N - k + 1):
            if x[i - 1] > x[i - k - 1] and x[i - 1] > x[i + k - 1]:
                m_max[k - 1, i - 1] = True
            if x[i - 1] < x[i - k - 1] and x[i - 1] < x[i + k - 1]:
                m_min[k - 1, i - 1] = True

    # Step 2: find the scale with the most local maxima (or local minima)
    # - row-wise summation (i.e. sum each row)
    gamma_max = np.sum(m_max, axis=1)
    # the "axis=1" option makes it row-wise
    gamma_min = np.sum(m_min, axis=1)
    # - find scale with the most local maxima (or local minima)
    lambda_max = np.argmax(gamma_max)
    lambda_min = np.argmax(gamma_min)

    # Step 3: Use lambda to remove all elements of m for which k>lambda
    m_max = m_max[: (lambda_max + 1), :]
    m_min = m_min[: (lambda_min + 1), :]

    # Step 4: Find peaks (and onsets)
    # - column-wise summation
    m_max_sum = np.sum(m_max == False, axis=0)
    m_min_sum = np.sum(m_min == False, axis=0)
    peaks = np.asarray(np.where(m_max_sum == 0)).astype(int)
    onsets = np.asarray(np.where(m_min_sum == 0)).astype(int)

    if show:
        _, ax0 = plt.subplots(nrows=1, ncols=1, sharex=True)
        ax0.plot(signal, label="signal")
        ax0.scatter(peaks, signal[peaks], c="r")
        ax0.scatter(onsets, signal[onsets], c="b")
        ax0.set_title("PPG Peaks (Method by Bishop et al., 2018)")

    return peaks, onsets
