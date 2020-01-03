# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from ..signal import signal_smooth
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def ecg_findpeaks(ecg_cleaned, sampling_rate=1000, method="neurokit",
                  enable_plot=False):
    """Find R-peaks in an ECG signal.

    Find R-peaks in an ECG signal using the specified method.

    Parameters
    ----------
    ecg_cleaned : list, array or Series
        The cleaned ECG channel as returned by `ecg_clean()`.
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
        Defaults to 1000.
    method : string
        The algorithm to be used for R-peak detection. Defaults to "neurokit".
    enable_plot : boolean
        Visualize the thresholds used in the algorithm specified by `method`.
        Defaults to False.

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as the input signal in which occurences of
        R-peaks marked as "1" in a list of zeros with the same length as
        `ecg_cleaned`. Accessible with the keys "ECG_Peaks".
    info : dict
        A dictionary containing additional information, in this case the
        samples at which R-peaks occur, accessible with the key "ECG_Peaks".

    See Also
    --------
    ecg_clean, ecg_rate, ecg_process, ecg_plot

    Examples
    --------
    >>>
    """
    # Determine method and search R-peaks.
    peakfun = False
    method = method.lower()
    if method == "neurokit":
        peakfun = _ecg_findpeaks_nk
    elif method == "pamtompkins":
        peakfun = _ecg_findpeaks_pantompkins
    if not peakfun:
        print("NeuroKit error: Please choose a valid method.")

    peaks = peakfun(signal=ecg_cleaned, sampling_rate=sampling_rate,
                    enable_plot=enable_plot)

    # Prepare output.
    peaks_signal = np.zeros(len(ecg_cleaned))
    peaks_signal[peaks] = 1
    signals = pd.DataFrame({"ECG_Peaks": peaks_signal})

    info = {"ECG_Peaks": peaks}

    return(signals, info)


def _ecg_findpeaks_pantompkins():
    # TODO
    pass


def _ecg_findpeaks_nk(signal, sampling_rate, smoothwindow=.1, avgwindow=.75,
                      gradthreshweight=1.5, minlenweight=0.4, mindelay=0.3,
                      enable_plot=False):
    """
    All tune-able parameters are specified as keyword arguments. The `signal`
    must be the highpass-filtered raw ECG with a lowcut of .5 Hz.
    """
    if enable_plot is True:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    # Compute the ECG's gradient as well as the gradient threshold. Run with
    # enable_plot=True in order to get an idea of the threshold.
    grad = np.gradient(signal)
    absgrad = np.abs(grad)
    smooth_kernel = int(np.rint(smoothwindow * sampling_rate))
    avg_kernel = int(np.rint(avgwindow * sampling_rate))
    smoothgrad = signal_smooth(absgrad, kernel="boxcar", size=smooth_kernel)
    avggrad = signal_smooth(smoothgrad, kernel="boxcar", size=avg_kernel)
    gradthreshold = gradthreshweight * avggrad
    mindelay = int(np.rint(sampling_rate * mindelay))

    if enable_plot is True:
        ax1.plot(signal)
        ax2.plot(smoothgrad)
        ax2.plot(gradthreshold)

    # Identify start and end of QRS complexes.
    qrs = smoothgrad > gradthreshold
    beg_qrs = np.where(np.logical_and(np.logical_not(qrs[0:-1]), qrs[1:]))[0]
    end_qrs = np.where(np.logical_and(qrs[0:-1], np.logical_not(qrs[1:])))[0]
    # Throw out QRS-ends that precede first QRS-start.
    end_qrs = end_qrs[end_qrs > beg_qrs[0]]

    # Identify R-peaks within QRS (ignore QRS that are too short).
    num_qrs = min(beg_qrs.size, end_qrs.size)
    min_len = np.mean(end_qrs[:num_qrs] - beg_qrs[:num_qrs]) * minlenweight
    peaks = [0]

    for i in range(num_qrs):

        beg = beg_qrs[i]
        end = end_qrs[i]
        len_qrs = end - beg

        if len_qrs < min_len:
            continue

        if enable_plot is True:
            ax2.axvspan(beg, end, facecolor="m", alpha=0.5)

        # Find local maxima and their prominence within QRS.
        data = signal[beg:end]
        locmax, props = find_peaks(data, prominence=(None, None))

        if locmax.size > 0:
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            # Enforce minimum delay between peaks.
            if peak - peaks[-1] > mindelay:
                peaks.append(peak)

    peaks.pop(0)

    if enable_plot is True:
        ax1.scatter(peaks, signal[peaks], c="r")

    return np.asarray(peaks).astype(int)
