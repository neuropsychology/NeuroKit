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
        The algorithm to be used for R-peak detection. Can be one of 'neurokit' (default),
        or 'pamtompkins1985'.
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
    >>>
    >>> # Different methods
    >>> _, neurokit = nk.ecg_findpeaks(cleaned, method="neurokit")
    >>> _, pantompkins1985 = nk.ecg_findpeaks(cleaned, method="pantompkins1985")
    >>> _, gamboa2008 = nk.ecg_findpeaks(cleaned, method="gamboa2008")
    >>> _, ssf = nk.ecg_findpeaks(cleaned, method="ssf")
    >>> nk.events_plot([neurokit["ECG_Peaks"],
                        pantompkins1985["ECG_Peaks"],
                        gamboa2008["ECG_Peaks"],
                        ssf["ECG_Peaks"]], cleaned)

    References
    --------------
    - Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci and electrophysiology. PhD ThesisUniversidade.
    - Jiapu Pan and Willis J. Tompkins. A Real-Time QRS Detection Algorithm.
    In: IEEE Transactions on Biomedical Engineering BME-32.3 (1985), pp. 230–236.
    - W. Zong, T. Heldt, G.B. Moody, and R.G. Mark. An open-source algorithm to detect onset of arterial blood pressure pulses. In Computers in
Cardiology, 2003, pages 259–262, 2003.
    """
    # Try retrieving right column
    if isinstance(ecg_cleaned, pd.DataFrame):
        try:
            ecg_cleaned = ecg_cleaned["ECG_Clean"]
        except NameError:
            try:
                ecg_cleaned = ecg_cleaned["ECG_Raw"]
            except NameError:
                ecg_cleaned = ecg_cleaned["ECG"]


    method = method.lower()  # remove capitalised letters
    # Run peak detection algorithm
    if method in ["nk", "nk2", "neurokit", "neurokit2"]:
        rpeaks = _ecg_findpeaks_neurokit(ecg_cleaned, sampling_rate)
    elif method in ["pantompkins", "pantompkins1985"]:
        rpeaks = _ecg_findpeaks_pantompkins(ecg_cleaned, sampling_rate)
    elif method in ["gamboa2008", "gamboa"]:
        rpeaks = _ecg_findpeaks_gamboa(ecg_cleaned, sampling_rate)
    elif method in ["ssf", "slopesumfunction", "zong", "zong2003"]:
        rpeaks = _ecg_findpeaks_ssf(ecg_cleaned, sampling_rate)
    else:
        raise ValueError("NeuroKit error: ecg_findpeaks(): 'method' should be "
                         "one of 'neurokit' or 'pamtompkins'.")


    # Prepare output.
    info = {"ECG_Peaks": rpeaks}
    signals = _signals_from_peakinfo(info, peak_indices=info["ECG_Peaks"], length=len(ecg_cleaned))

    return signals, info





# =============================================================================
# NeuroKit
# =============================================================================
def _ecg_findpeaks_neurokit(signal, sampling_rate=1000, smoothwindow=.1, avgwindow=.75,
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




# =============================================================================
# Pan & Tompkins (1985)
# =============================================================================
def _ecg_findpeaks_pantompkins(signal, sampling_rate=1000):
    """
    From https://github.com/berndporr/py-ecg-detectors/

    - Jiapu Pan and Willis J. Tompkins. A Real-Time QRS Detection Algorithm.
    In: IEEE Transactions on Biomedical Engineering BME-32.3 (1985), pp. 230–236.
    """
    diff = np.diff(signal)

    squared = diff*diff

    N = int(0.12*sampling_rate)
    mwa = _ecg_findpeaks_MWA(squared, N)
    mwa[:int(0.2*sampling_rate)] = 0

    mwa_peaks = _ecg_findpeaks_peakdetect(mwa, sampling_rate)

    return mwa_peaks






# =============================================================================
# Gamboa (2008)
# =============================================================================

def _ecg_findpeaks_gamboa(signal, sampling_rate=1000, tol=0.002):
    """
    From https://github.com/PIA-Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L834

    - Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci and electrophysiology. PhD ThesisUniversidade.
    """

    # convert to samples
    v_100ms = int(0.1 * sampling_rate)
    v_300ms = int(0.3 * sampling_rate)
    hist, edges = np.histogram(signal, 100, density=True)

    TH = 0.01
    F = np.cumsum(hist)

    v0 = edges[np.nonzero(F > TH)[0][0]]
    v1 = edges[np.nonzero(F < (1 - TH))[0][-1]]

    nrm = max([abs(v0), abs(v1)])
    norm_signal = signal / float(nrm)

    d2 = np.diff(norm_signal, 2)

    b = np.nonzero((np.diff(np.sign(np.diff(-d2)))) == -2)[0] + 2
    b = np.intersect1d(b, np.nonzero(-d2 > tol)[0])

    if len(b) < 3:
        rpeaks = []
    else:
        b = b.astype('float')
        rpeaks = []
        previous = b[0]
        for i in b[1:]:
            if i - previous > v_300ms:
                previous = i
                rpeaks.append(np.argmax(signal[int(i):int(i + v_100ms)]) + i)

    rpeaks = sorted(list(set(rpeaks)))
    rpeaks = np.array(rpeaks, dtype='int')
    return rpeaks





# =============================================================================
# Slope Sum Function (SSF) - Zong et al. (2003)
# =============================================================================

def _ecg_findpeaks_ssf(signal, sampling_rate=1000, threshold=20, before=0.03, after=0.01):
    """
    From https://github.com/PIA-Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L448

    - W. Zong, T. Heldt, G.B. Moody, and R.G. Mark. An open-source algorithm to detect onset of arterial blood pressure pulses. In Computers in
Cardiology, 2003, pages 259–262, 2003.
    """
    # TODO: Doesn't really seems to work

    # convert to samples
    winB = int(before * sampling_rate)
    winA = int(after * sampling_rate)

    Rset = set()
    length = len(signal)

    # diff
    dx = np.diff(signal)
    dx[dx >= 0] = 0
    dx = dx ** 2

    # detection
    idx, = np.nonzero(dx > threshold)
    idx0 = np.hstack(([0], idx))
    didx = np.diff(idx0)

    # search
    sidx = idx[didx > 1]
    for item in sidx:
        a = item - winB
        if a < 0:
            a = 0
        b = item + winA
        if b > length:
            continue

        r = np.argmax(signal[a:b]) + a
        Rset.add(r)

    # output
    rpeaks = list(Rset)
    rpeaks.sort()
    rpeaks = np.array(rpeaks, dtype='int')
    return rpeaks


# =============================================================================
# Utilities
# =============================================================================

def _ecg_findpeaks_MWA(signal, window_size):
    """
    From https://github.com/berndporr/py-ecg-detectors/
    """
    mwa = np.zeros(len(signal))
    for i in range(len(signal)):
        if i < window_size:
            section = signal[0:i]
        else:
            section = signal[i-window_size:i]

        if i!=0:
            mwa[i] = np.mean(section)
        else:
            mwa[i] = signal[i]

    return mwa






def _ecg_findpeaks_peakdetect(detection, sampling_rate=1000):
    """
    From https://github.com/berndporr/py-ecg-detectors/
    """
    min_distance = int(0.25 * sampling_rate)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(len(detection)):

        if i > 0 and i < len(detection) - 1:
            if detection[i-1] < detection[i] and detection[i+1] < detection[i]:
                peak = i
                peaks.append(i)

                if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.3 * sampling_rate:

                    signal_peaks.append(peak)
                    indexes.append(index)
                    SPKI = 0.125 * detection[signal_peaks[-1]] + 0.875 * SPKI
                    if RR_missed!=0:
                        if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                            missed_section_peaks = peaks[indexes[-2] + 1:indexes[-1]]
                            missed_section_peaks2 = []
                            for missed_peak in missed_section_peaks:
                                if missed_peak - signal_peaks[-2] > min_distance and signal_peaks[-1] - missed_peak > min_distance and detection[missed_peak] > threshold_I2:
                                    missed_section_peaks2.append(missed_peak)

                            if len(missed_section_peaks2) > 0:
                                missed_peak = missed_section_peaks2[np.argmax(detection[missed_section_peaks2])]
                                missed_peaks.append(missed_peak)
                                signal_peaks.append(signal_peaks[-1])
                                signal_peaks[-2] = missed_peak

                else:
                    noise_peaks.append(peak)
                    NPKI = 0.125 * detection[noise_peaks[-1]] + 0.875 * NPKI

                threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
                threshold_I2 = 0.5 * threshold_I1

                if len(signal_peaks) > 8:
                    RR = np.diff(signal_peaks[-9:])
                    RR_ave = int(np.mean(RR))
                    RR_missed = int(1.66 * RR_ave)

                index = index + 1

    signal_peaks.pop(0)

    return signal_peaks
