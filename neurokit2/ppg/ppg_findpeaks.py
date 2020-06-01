# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from ..signal import signal_formatpeaks, signal_smooth


def ppg_findpeaks(ppg_cleaned, sampling_rate=1000, method="elgendi", show=False):
    """
    Find systolic peaks in a photoplethysmogram (PPG) signal.

    Parameters
    ----------
    ppg_cleaned : list, array or Series
        The cleaned PPG channel as returned by `ppg_clean()`.
    sampling_rate : int
        The sampling frequency of the PPG (in Hz, i.e., samples/second). The
        default is 1000.
    method : str
        The processing pipeline to apply. Can be one of "elgendi". The default
        is "elgendi".
    show : bool
        If True, returns a plot of the thresholds used during peak detection.
        Useful for debugging. The default is False.

    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the
        samples at which systolic peaks occur, accessible with the key
        "PPG_Peaks".

    See Also
    --------
    ppg_simulate, ppg_clean

    Examples
    --------
    >>> import neurokit2 as nk
    >>> import matplotlib.pyplot as plt
    >>>
    >>> ppg = nk.ppg_simulate(heart_rate=75, duration=30)
    >>> ppg_clean = nk.ppg_clean(ppg)
    >>> info = nk.ppg_findpeaks(ppg_clean)
    >>> peaks = info["PPG_Peaks"]
    >>>
    >>> plt.plot(ppg, label="raw PPG") #doctest: +SKIP
    >>> plt.plot(ppg_clean, label="clean PPG") #doctest: +SKIP
    >>> plt.scatter(peaks, ppg[peaks], c="r", label="systolic peaks") #doctest: +SKIP
    >>> plt.legend() #doctest: +SKIP

    References
    ----------
    - Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic
    Peak Detection in Acceleration Photoplethysmograms Measured from Emergency
    Responders in Tropical Conditions. PLoS ONE 8(10): e76585.
    doi:10.1371/journal.pone.0076585.

    """
    method = method.lower()
    if method in ["elgendi"]:
        peaks = _ppg_findpeaks_elgendi(ppg_cleaned, sampling_rate, show=show)
    else:
        raise ValueError("Neurokit error: Please use one of the following" " methods: 'elgendi'.")

    # Prepare output.
    info = {"PPG_Peaks": peaks}

    return info


def _ppg_findpeaks_elgendi(
    signal, sampling_rate=1000, peakwindow=0.111, beatwindow=0.667, beatoffset=0.02, mindelay=0.3, show=False
):
    """
    Implementation of Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection in
    Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical Conditions. PLoS ONE 8(10): e76585.
    doi:10.1371/journal.pone.0076585.

    All tune-able parameters are specified as keyword arguments. `signal` must be the bandpass-filtered raw PPG
    with a lowcut of .5 Hz, a highcut of 8 Hz.

    """
    if show:
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax0.plot(signal, label="filtered")

    # Ignore the samples with negative amplitudes and square the samples with
    # values larger than zero.
    signal[signal < 0] = 0
    sqrd = signal ** 2

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
        ax0.scatter(peaks, signal[peaks], c="r")

    peaks = np.asarray(peaks).astype(int)
    return peaks
