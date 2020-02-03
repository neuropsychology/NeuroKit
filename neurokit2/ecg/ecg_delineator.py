# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

def ecg_delineator(rpeaks, sampling_rate=500):
    """Delineate QRS complex.
    Function to delineate the QRS complex.

    Parameters
    ----------
    rpeaks : list, array or Series
        The samples at which R-peaks occur.
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
        Defaults to 500.

    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the
        samples at which P-peaks and T-peaks occur, accessible with the key "ECG_P_Peaks" and "ECG_R_Peaks" respectively.

    See Also
    --------
    ecg_clean, ecg_fixpeaks, ecg_peaks, ecg_rate, ecg_process, ecg_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> ecg = nk.ecg_simulate(duration=10, sampling_rate=500)
    >>> cleaned = nk.ecg_clean(ecg, sampling_rate=1000)
    >>> info1 = nk.ecg_findpeaks(cleaned)
    >>> rpeaks = info1["ECG_R_Peaks"]
    >>> info2 = nk.ecg_delineator(rpeaks, sampling_rate=500)
    >>> nk.events_plot(info2["ECG_P_Peaks"], cleaned)
    >>> nk.events_plot(info2["ECG_T_Peaks"], cleaned)

    References
    --------------
    - Martínez, J. P., Almeida, R., Olmos, S., Rocha, A. P., & Laguna, P. (2004). A wavelet-based ECG delineator: evaluation on standard databases. IEEE Transactions on biomedical engineering, 51(4), 570-581.

    """
    # Try loading pywt
    try:
        import pywt
    except ImportError:
        raise ImportError("NeuroKit error: ecg_delineator(): the 'PyWavelets' "
                          "module is required for this method to run. ",
                          "Please install it first (`pip install PyWavelets`).")

    # P-Peaks and T-Peaks
    ppeaks, tpeaks = _ecg_peaks_delineator(rpeaks, sampling_rate)
    info = {"ECG_P_Peaks":ppeaks,
            "ECG_T_Peaks":tpeaks}
    return info


def _ecg_peaks_delineator(rpeaks, sampling_rate=500):
    # first derivative of the Gaissian signal
    scales = np.array([1, 2, 4, 8, 16])
    cwtmatr, freqs = pywt.cwt(ecg, scales, 'gaus1', sampling_period=1.0/sampling_rate)

    # search between R-peaks for significant peaks in cwt of scales 2^4
    keep_peaks = []
    for i in range(len(rpeaks)-1):
        search_window = cwtmatr[4,rpeaks[i]:rpeaks[i+1]]
        height = 0.125*np.sqrt(np.mean(np.square(search_window)))
        peaks, heights = scipy.signal.find_peaks(np.abs(search_window), height=height)
        peaks = peaks + rpeaks[i]
        threshold1 = 0.125*max(search_window) # min height of peaks
        threshold2 = 0.8*max(search_window) # max height of peaks
        significant_index = [j for j in range(len(peaks))
                             if heights["peak_heights"][j] > threshold1
                             and heights["peak_heights"][j] < threshold2]

        significant_peaks = []
        for index in significant_index:
            significant_peaks.append(peaks[index])
        keep_peaks = np.concatenate((keep_peaks, significant_peaks), axis=0)
    keep_peaks = np.int64(keep_peaks)

    # locate P-peaks and T-peaks
#    max_wv_peak_dist = int(0.1 * sampling_rate)
    peaks = []
    for index_cur, index_next in zip(keep_peaks[:-1], keep_peaks[1:]):
        # look for a pair of negative-positive maxima
        correct_sign = cwtmatr[4,:][index_cur] < 0 and cwtmatr[4,:][index_next] > 0
#       near = (index_next - index_cur) < max_wv_peak_dist #limit 2
#       if near and correct_sign:
        if correct_sign:
            peaks.append(nk.signal_zerocrossings(
                    cwtmatr[4,:][index_cur:index_next])[0] + index_cur)

    # delineate T P peaks
    tpeaks = []
    ppeaks = []
    for i in range(len(peaks)):
        rpeaks_distance = abs(peaks[i] - rpeaks)
        rpeaks_closest = rpeaks[np.argmin(rpeaks_distance)]
        if (rpeaks_closest - peaks[i]) > 0:
            ppeaks.append(peaks[i])
        elif (rpeaks_closest - peaks[i]) < 0:
            tpeaks.append(peaks[i])
    ppeaks = np.int64(ppeaks)
    tpeaks = np.int64(tpeaks)
    return ppeaks, tpeaks
