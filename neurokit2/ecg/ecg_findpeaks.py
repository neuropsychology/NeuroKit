# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

from ..signal import signal_smooth
from ..signal.signal_from_indices import _signals_from_peakinfo



def ecg_findpeaks(ecg_cleaned, sampling_rate=1000, method="neurokit", show=False):
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
    show : bool
        If True, will return a plot to visualizing the thresholds used in the
        algorithm. Useful for debugging.

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
    >>> import neurokit2 as nk
    >>>
    >>> ecg = nk.ecg_simulate(duration=10, sampling_rate=1000)
    >>> cleaned = nk.ecg_clean(ecg, sampling_rate=1000)
    >>> signals, info = nk.ecg_findpeaks(cleaned)
    >>> nk.events_plot(info["ECG_Peaks"], cleaned)
    """
    # Try retrieving right column
    if isinstance(ecg_cleaned, pd.DataFrame):
        try:
            ecg_cleaned = ecg_cleaned["RSP_Clean"]
        except NameError:
            try:
                ecg_cleaned = ecg_cleaned["ECG_Raw"]
            except NameError:
                ecg_cleaned = ecg_cleaned["RSP"]

    method = method.lower()  # remove capitalised letters

    # Run peak detection algorithm
    if method == "neurokit":
        rpeaks = _ecg_findpeaks_neurokit(ecg_cleaned,
                                        sampling_rate,
                                        smoothwindow=.1,
                                        avgwindow=.75,
                                        gradthreshweight=1.5,
                                        minlenweight=0.4,
                                        mindelay=0.3,
                                        show=show)
    elif method == "pamtompkins":
        rpeaks = _ecg_findpeaks_pantompkins()
    else:
        raise ValueError("NeuroKit error: ecg_findpeaks(): 'method' should be "
                         "one of 'neurokit' or 'pamtompkins'.")


    # Prepare output.
    info={"ECG_Peaks": rpeaks}
    signals = _signals_from_peakinfo(info, peak_indices=info["ECG_Peaks"], length=len(ecg_cleaned))

    return signals, info



# =============================================================================
# Pantompkins
# =============================================================================
def _ecg_findpeaks_pantompkins():
    raise ValueError("NeuroKit error: ecg_findpeaks(): pamtompkins 'method' "
                     "is not implemented yet.")



# =============================================================================
# NeuroKit
# =============================================================================
def _ecg_findpeaks_neurokit(signal, sampling_rate, smoothwindow=.1, avgwindow=.75,
                            gradthreshweight=1.5, minlenweight=0.4, mindelay=0.3,
                            show=False):
    """
    All tune-able parameters are specified as keyword arguments. The `signal`
    must be the highpass-filtered raw ECG with a lowcut of .5 Hz.
    """
    if show is True:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    # Compute the ECG's gradient as well as the gradient threshold. Run with
    # show=True in order to get an idea of the threshold.
    grad = np.gradient(signal)
    absgrad = np.abs(grad)
    smooth_kernel = int(np.rint(smoothwindow * sampling_rate))
    avg_kernel = int(np.rint(avgwindow * sampling_rate))
    smoothgrad = signal_smooth(absgrad, kernel="boxcar", size=smooth_kernel)
    avggrad = signal_smooth(smoothgrad, kernel="boxcar", size=avg_kernel)
    gradthreshold = gradthreshweight * avggrad
    mindelay = int(np.rint(sampling_rate * mindelay))

    if show is True:
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

        if show is True:
            ax2.axvspan(beg, end, facecolor="m", alpha=0.5)

        # Find local maxima and their prominence within QRS.
        data = signal[beg:end]
        locmax, props = scipy.signal.find_peaks(data, prominence=(None, None))

        if locmax.size > 0:
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            # Enforce minimum delay between peaks.
            if peak - peaks[-1] > mindelay:
                peaks.append(peak)

    peaks.pop(0)

    if show is True:
        ax1.scatter(peaks, signal[peaks], c="r")

    peaks = np.asarray(peaks).astype(int)  # Convert to int
    return peaks
