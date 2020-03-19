# -*- coding: utf-8 -*-
import numpy as np
import biosppy


def ecg_rsa(rsp_signals, rpeaks, sampling_rate=1000):
    """
    Returns Respiratory Sinus Arrhythmia (RSA) features. Only the Peak-to-trough (P2T) algorithm is currently implemented (see details).

    Parameters
    ----------
    rsp_signals : DataFrame
        DataFrame obtained from `rsp_process()`.
    rpeaks : dict
        The samples at which the R-peaks of the ecg signal occur. Dict returned by
        `ecg_peaks()` or `ecg_process()`.
    sampling_rate : int
        The sampling frequency of `rsp_signal` (in Hz, i.e., samples/second).

    Returns
    ----------
    rsa : dict
        Contains RSA features.

    Example
    ----------
    >>> import neurokit as nk
    >>>
    >>> data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/example_bio_100hz.csv")
    >>>
    >>> # Process the data
    >>> rsp_signals, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"], sampling_rate=100)
    >>> rsa = nk.ecg_rsa(rsp_signals, info)

    Returns
    -------
    rsa : dict
        A dictionary containing information about the RSA features.

    References
    ------------
    - Lewis, G. F., Furman, S. A., McCool, M. F., & Porges, S. W. (2012). Statistical strategies to quantify respiratory sinus arrhythmia: Are commonly used metrics equivalent?. Biological psychology, 89(2), 349-364.
    """
    # Extract cycles
    rpeaks = rpeaks["ECG_R_Peaks"]
    rsp_cycles = _ecg_rsa_cycles(rsp_signals)
    rsp_onsets = rsp_cycles["RSP_Inspiration_Onsets"]
    rsp_cycle_center = rsp_cycles["RSP_Expiration_Onsets"]
    rsp_cycle_center = np.array(rsp_cycle_center)[rsp_cycle_center > rsp_onsets[0]]

    if len(rsp_cycle_center) - len(rsp_onsets) == 0:
        rsp_cycle_center = rsp_cycle_center[:-1]
    if len(rsp_cycle_center) - len(rsp_onsets) != -1:
        print("NeuroKit Error: ecg_rsp(): Couldn't find clean rsp cycles onsets and centers. Check your RSP signal.")

    rsa = {}

    # Peak-to-trough algorithm (P2T)
    # ===============================
    # Find all RSP cycles and the Rpeaks within
    cycles_rri = []
    for idx in range(len(rsp_onsets) - 1):
        cycle_init = rsp_onsets[idx]
        cycle_end = rsp_onsets[idx + 1]
        cycles_rri.append(rpeaks[np.logical_and(rpeaks >= cycle_init,
                                                rpeaks < cycle_end)])

    # Iterate over all cycles
    rsa["RSA_P2T_Values"] = []
    for cycle in cycles_rri:
    # Estimate of RSA during each breath
        RRis = np.diff(cycle) / sampling_rate
        if len(RRis) > 1:
            rsa["RSA_P2T_Values"].append(np.max(RRis) - np.min(RRis)) # P2T in ms
        else:
            rsa["RSA_P2T_Values"].append(np.nan)
    rsa["RSA_P2T_Mean"] = pd.Series(rsa["RSA_P2T_Values"]).mean()
    rsa["RSA_P2T_Mean_log"] = np.log(rsa["RSA_P2T_Mean"])
    rsa["RSA_P2T_Variability"] = pd.Series(rsa["RSA_P2T_Values"]).std()

    return(rsa)


# =============================================================================
# Internals
# =============================================================================
def _ecg_rsa_cycles(rsp_signals):
    """
    Extract respiratory cycles.
    """
    inspiration_onsets = np.intersect1d(np.where(rsp_signals["RSP_Phase"] == 1)[0], np.where(rsp_signals["RSP_PhaseCompletion"] == 0)[0], assume_unique=True)

    expiration_onsets = np.intersect1d(np.where(rsp_signals["RSP_Phase"] == 0)[0], np.where(rsp_signals["RSP_PhaseCompletion"] == 0)[0], assume_unique=True)

    cycles_length = np.diff(inspiration_onsets)

    rsp_cycles = {"RSP_Inspiration_Onsets": inspiration_onsets,
                  "RSP_Expiration_Onsets": expiration_onsets,
                  "RSP_Cycles_Length": cycles_length}

    return(rsp_cycles)
