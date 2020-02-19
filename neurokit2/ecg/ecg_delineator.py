# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#from ..signal import signal_zerocrossings
#from ..signal import signal_detrend
#from ..signal import signal_smooth
#from ..signal import signal_filter


def ecg_delineator(ecg, rpeaks, sampling_rate=500, cleaning=False):
    """Delineate QRS complex.
    Function to delineate the QRS complex.

    Parameters
    ----------
    ecg : list, array or Series
        The raw ECG signal.
    rpeaks : list, array or Series
        The samples at which R-peaks occur. Accessible with the key "ECG_R_Peaks" in the info dictionary returned by `ecg_findpeaks()`.
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
    >>> rpeaks = nk.ecg_findpeaks(cleaned)
    >>> info2 = nk.ecg_delineator(cleaned, rpeaks, sampling_rate=500)
    >>> nk.events_plot(info2["ECG_P_Peaks"], cleaned)
    >>> nk.events_plot(info2["ECG_T_Peaks"], cleaned)

    References
    --------------
    - Martínez, J. P., Almeida, R., Olmos, S., Rocha, A. P., & Laguna, P. (2004). A wavelet-based ECG delineator: evaluation on standard databases. IEEE Transactions on biomedical engineering, 51(4), 570-581.

    """
    # Try retrieving right column
    if isinstance(rpeaks, dict):
        rpeaks = rpeaks["ECG_R_Peaks"]

    # P-Peaks and T-Peaks
    tpeaks, ppeaks = _peaks_delineator(ecg, rpeaks, sampling_rate)

    # qrs onsets and offsets
    qrs_onsets, qrs_offsets = _onset_offset_delineator(rpeaks,
                                                       peak_type="rpeaks",
                                                       sampling_rate=sampling_rate)

    # ppeaks onsets and offsets
    p_onsets, p_offsets = _onset_offset_delineator(ppeaks,
                                                   peak_type="ppeaks",
                                                   sampling_rate=sampling_rate)

    info = {"ECG_P_Peaks": ppeaks,
            "ECG_T_Peaks": tpeaks,
            "ECG_R_Onsets": qrs_onsets,
            "ECG_R_Offsets": qrs_offsets,
            "ECG_P_Onsets": p_onsets,
            "ECG_P_Offsets": p_offsets}
    return info


def _onset_offset_delineator(peaks, peak_type="rpeaks", sampling_rate=500):
    # Try loading pywt
    try:
        import pywt
    except ImportError:
        raise ImportError("NeuroKit error: ecg_delineator(): the 'PyWavelets' "
                          "module is required for this method to run. ",
                          "Please install it first (`pip install PyWavelets`).")
    # first derivative of the Gaissian signal
    scales = np.array([1, 2, 4, 8, 16])
    cwtmatr, freqs = pywt.cwt(ecg, scales, 'gaus1', sampling_period=1.0/sampling_rate)

    half_wave_width = int(0.1*sampling_rate)  # NEED TO CHECK
    onsets = []
    offsets = []
    for index_peak in peaks:
        # find onset
        if peak_type == "rpeaks":
            search_window = cwtmatr[2, index_peak - half_wave_width: index_peak]
            prominence = 0.20*max(search_window)
            height = 0.0
            wt_peaks, wt_peaks_data = find_peaks(search_window, height=height,
                                       prominence=prominence)
        elif peak_type == "tpeaks" or peak_type == "ppeaks":
            search_window = - cwtmatr[4, index_peak - half_wave_width: index_peak]
            prominence = 0.50*max(search_window)
            height = 0.0
            wt_peaks, wt_peaks_data = find_peaks(search_window, height=height,
                                           prominence=prominence)
        if len(wt_peaks) == 0:
            print("Fail to find onset at index: %d", index_peak)
            continue
        # The last peak is nfirst in (Martinez, 2004)
        nfirst = wt_peaks[-1] + index_peak - half_wave_width
        if peak_type == "rpeaks":
            if wt_peaks_data['peak_heights'][-1] > 0:
                epsilon_onset = 0.05 * wt_peaks_data['peak_heights'][-1]
            elif wt_peaks_data['peak_heights'][-1] > 0:
                epsilon_onset = 0.07 * wt_peaks_data['peak_heights'][-1]
        elif peak_type == "ppeaks":
            epsilon_onset = 0.50 * wt_peaks_data['peak_heights'][-1]
        elif peak_type == "tpeaks":
            epsilon_onset = 0.25 * wt_peaks_data['peak_heights'][-1]
        leftbase = wt_peaks_data['left_bases'][-1] + index_peak - half_wave_width
        if peak_type == "rpeaks":
            candidate_onsets = np.where(cwtmatr[2, nfirst-100: nfirst] <
                                        epsilon_onset)[0] + nfirst - 100
        elif peak_type == "tpeaks" or peak_type == "ppeaks":
            candidate_onsets = np.where(-cwtmatr[4, nfirst-100: nfirst] <
                                        epsilon_onset)[0] + nfirst - 100
        candidate_onsets = candidate_onsets.tolist() + [leftbase]
        onsets.append(max(candidate_onsets))

        # find offset
        if peak_type == "rpeaks":
            search_window = - cwtmatr[2, index_peak: index_peak + half_wave_width]
            prominence = 0.5*max(search_window)
            wt_peaks, wt_peaks_data = find_peaks(search_window, height=height,
                                                 prominence=prominence)
        elif peak_type == "tpeaks" or peak_type == "ppeaks":
            search_window =  cwtmatr[4, index_peak: index_peak + half_wave_width]
            prominence = 0.5*max(search_window)
            wt_peaks, wt_peaks_data = find_peaks(search_window, height=height,
                                           prominence=prominence)
        if len(wt_peaks) == 0:
            print("Fail to find offset at index: %d", index_peak)
            continue
        nlast = wt_peaks[0] + index_peak
        if peak_type == "rpeaks":
            if wt_peaks_data['peak_heights'][0] > 0:
                epsilon_offset = 0.125 * wt_peaks_data['peak_heights'][0]
            elif wt_peaks_data['peak_heights'][0] > 0:
                epsilon_offset = 0.71 * wt_peaks_data['peak_heights'][0]
        elif peak_type == "ppeaks":
            epsilon_offset = 0.9 * wt_peaks_data['peak_heights'][0]
        elif peak_type == "tpeaks":
            epsilon_offset = 0.4 * wt_peaks_data['peak_heights'][0]
        rightbase = wt_peaks_data['right_bases'][0] + index_peak
        if peak_type == "rpeaks":
            candidate_offsets = np.where((-cwtmatr[2, nlast: nlast + 100]) <
                                         epsilon_offset)[0] + nlast
        elif peak_type == "tpeaks" or peak_type == "ppeaks":
            candidate_offsets = np.where((cwtmatr[4, nlast: nlast + 100]) <
                                         epsilon_offset)[0] + nlast
        candidate_offsets = candidate_offsets.tolist() + [rightbase]
        offsets.append(min(candidate_offsets))

    return onsets, offsets

def _peaks_delineator(ecg, rpeaks, cleaning=False, sampling_rate=500):
    # Try loading pywt
    try:
        import pywt
    except ImportError:
        raise ImportError("NeuroKit error: ecg_delineator(): the 'PyWavelets' "
                          "module is required for this method to run. ",
                          "Please install it first (`pip install PyWavelets`).")
    # first derivative of the Gaissian signal
    scales = np.array([1, 2, 4, 8, 16])
    cwtmatr, freqs = pywt.cwt(ecg, scales, 'gaus1', sampling_period=1.0/sampling_rate)

    if cleaning is not False:
        ecg_clean = _cleaning(ecg, sampling_rate=sampling_rate, lowpass=True)
        scales = np.array([1, 2, 4, 8, 16])
        cwtmatr_clean, freqs_clean = pywt.cwt(ecg_clean, scales, 'gaus1', sampling_period=1.0/sampling_rate)

    qrs_duration = 0.1

    search_boundary = int(0.9 * qrs_duration * sampling_rate / 2)
    significant_peaks_groups = []
    tppeaks_pairs = []
    tppeaks = []
    for i in range(len(rpeaks)-1):
        # search for T peaks and P peaks from R peaks
        start = rpeaks[i] + search_boundary
        end = rpeaks[i + 1] - search_boundary
        if cleaning is not False:
            search_window = cwtmatr_clean[4, start:end]
        else:
            search_window = cwtmatr[4, start:end]
        height = 0.25*np.sqrt(np.mean(np.square(search_window)))
        peaks_tp, heights_tp = find_peaks(np.abs(search_window), height=height)
        peaks_tp = peaks_tp + rpeaks[i] + search_boundary
        # set threshold for heights of peaks to find significant peaks in wavelet
        threshold = 0.125*max(search_window)
        significant_index = []
        significant_index = [j for j in range(len(peaks_tp)) if
                             heights_tp["peak_heights"][j] > threshold]
        significant_peaks_tp = []
        for index in significant_index:
            significant_peaks_tp.append(peaks_tp[index])
        significant_peaks_groups.append(_find_tppeaks(significant_peaks_tp, sampling_rate=sampling_rate))

    tpeaks, ppeaks = zip(*[(g[0], g[-1]) for g in significant_peaks_groups])
    return tpeaks, ppeaks



def _find_tppeaks(keep_tp, sampling_rate=500):
    # Try loading pywt
    try:
        import pywt
    except ImportError:
        raise ImportError("NeuroKit error: ecg_delineator(): the 'PyWavelets' "
                          "module is required for this method to run. ",
                          "Please install it first (`pip install PyWavelets`).")
    # first derivative of the Gaissian signal
    scales = np.array([1, 2, 4, 8, 16])
    cwtmatr, freqs = pywt.cwt(ecg, scales, 'gaus1', sampling_period=1.0/sampling_rate)
    max_search_duration = 0.05
    tppeaks = []
    for index_cur, index_next in zip(keep_tp[:-1], keep_tp[1:]):
        # limit 1
        correct_sign = cwtmatr[4, :][index_cur] < 0 and cwtmatr[4, :][index_next] > 0
    #    near = (index_next - index_cur) < max_wv_peak_dist #limit 2
    #    if near and correct_sign:
        if correct_sign:
            index_zero_cr = nk.signal_zerocrossings(
                cwtmatr[4, :][index_cur:index_next])[0] + index_cur
            nb_idx = int(max_search_duration * sampling_rate)
            index_max = np.argmax(ecg[index_zero_cr - nb_idx: index_zero_cr + nb_idx]) + (index_zero_cr - nb_idx)
            tppeaks.append(index_max)
    return tppeaks




def _cleaning(ecg, sampling_rate, lowpass=False, smooth=False):
    detrended = nk.signal_detrend(ecg, order=1)
    if lowpass:
        ecg_cleaned = nk.signal_filter(detrended, sampling_rate=sampling_rate, lowcut=2, highcut=12, method='butterworth')
        return ecg_cleaned
    elif smooth:
        ecg_smooth = nk.signal_smooth(ecg, size=sampling_rate*0.015)
        return ecg_smooth
    return detrended