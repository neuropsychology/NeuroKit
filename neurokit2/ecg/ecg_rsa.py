import pandas as pd
import numpy as np


#def ecg_rsa(rpeaks, rsp, sampling_rate=1000, method="P2T"):
#    """
#     Respiratory sinus arrhythmia (RSA) is a naturally occurring variation in heart rate that occurs during the breathing cycle, serving as a measure of parasympathetic nervous system activity. Neurophysiology informs us that the functional output of the myelinated vagus originating from the nucleus ambiguus has a respiratory rhythm. Thus, there would a temporal relation between the respiratory rhythm being expressed in the firing of these efferent pathways and the functional effect on the heart rate rhythm manifested as RSA. Several methods exist to quantify RSA, sich as the 'P2T'.
#
#     The peak to trough (P2T) method measures the statistical range in ms of the heart period oscillation associated with synchronous respiration. Operationally, subtracting the shortest heart period during inspiration from the longest heart period during a breath cycle produces an estimate of RSA during each breath. The peak-to-trough method makes no statistical assumption or correction (e.g., adaptive filtering) regarding other sources of variance in the heart period time series that may confound, distort, or interact with the metric such as slower periodicities and baseline trend. Although it has been proposed that the P2T method "acts as a time-domain filter dynamically centered at the exact ongoing respiratory frequency" (Grossman, 1992), the method does not transform the time series in any way, as a filtering process would. Instead the method uses knowledge of the ongoing respiratory cycle to associate segments of the heart period time series with either inhalation or exhalation (Lewis, 2012).
#
#    Parameters
#    ----------
#    rpeaks : list or ndarray
#        List of R peaks indices.
#    rsp : list or ndarray
#        Filtered RSP signal.
#    sampling_rate : int
#        Sampling rate (samples/second).
#    method : str
#        Only the Peak-to-trough (P2T) algorithm is currently implemented (see details).
#
#
#    Returns
#    ----------
#    rsa : dict
#        Contains Respiratory Sinus Arrhythmia (RSA) features.
#
#    Examples
#    ----------
#    >>> import neurokit2 as nk
#    >>> ec
#    >>> rsa = nk.ecg_rsa(rpeaks, rsp)
#
#
#    References
#    ------------
#    - Lewis, G. F., Furman, S. A., McCool, M. F., & Porges, S. W. (2012). Statistical strategies to quantify respiratory sinus arrhythmia: Are commonly used metrics equivalent?. Biological psychology, 89(2), 349-364.
#    """
#    # Preprocessing
#    # =================
#    rsp_cycles = rsp_find_cycles(rsp)
#    rsp_onsets = rsp_cycles["RSP_Cycles_Onsets"]
#    rsp_cycle_center = rsp_cycles["RSP_Expiration_Onsets"]
#    rsp_cycle_center = np.array(rsp_cycle_center)[rsp_cycle_center > rsp_onsets[0]]
#    if len(rsp_cycle_center) - len(rsp_onsets) == 0:
#        rsp_cycle_center = rsp_cycle_center[:-1]
#    if len(rsp_cycle_center) - len(rsp_onsets) != -1:
#        print("NeuroKit Error: ecg_rsp(): Couldn't find clean rsp cycles onsets and centers. Check your RSP signal.")
#        return()
#    rsa = {}
#
#
#    # Peak-to-trough algorithm (P2T)
#    # ===============================
#    # Find all RSP cycles and the Rpeaks within
#    cycles_rri = []
#    for idx in range(len(rsp_onsets) - 1):
#        cycle_init = rsp_onsets[idx]
#        cycle_end = rsp_onsets[idx + 1]
#        cycles_rri.append(rpeaks[np.logical_and(rpeaks >= cycle_init,
#                                                rpeaks < cycle_end)])
#
#    # Iterate over all cycles
#    rsa["RSA_P2T_Values"] = []
#    for cycle in cycles_rri:
#        RRis = np.diff(cycle)/sampling_rate
#        if len(RRis) > 1:
#            rsa["RSA_P2T_Values"].append(np.max(RRis) - np.min(RRis))
#        else:
#            rsa["RSA_P2T_Values"].append(np.nan)
#    rsa["RSA_P2T_Mean"] = pd.Series(rsa["RSA_P2T_Values"]).mean()
#    rsa["RSA_P2T_Mean_log"] = np.log(rsa["RSA_P2T_Mean"])
#    rsa["RSA_P2T_Variability"] = pd.Series(rsa["RSA_P2T_Values"]).std()
#
#    # Continuous RSA - Interpolation using a 3rd order spline
#    if len(rsp_cycle_center) - len(rsa["RSA_P2T_Values"]) != 0:
#        print("NeuroKit Error: ecg_rsp(): Couldn't find clean rsp cycles onsets and centers. Check your RSP signal.")
#        return()
#    values=pd.Series(rsa["RSA_P2T_Values"])
#    NaNs_indices = values.index[values.isnull()]  # get eventual artifacts indices
#    values = values.drop(NaNs_indices)  # remove the artifacts
#    value_times=(np.array(rsp_cycle_center))
#    value_times = np.delete(value_times, NaNs_indices)  # delete also the artifacts from times indices
#
#    rsa_interpolated = interpolate(values=values, value_times=value_times, sampling_rate=sampling_rate)
#
#
#    # Continuous RSA - Steps
#    current_rsa = np.nan
#
#    continuous_rsa = []
#    phase_counter = 0
#    for i in range(len(rsp)):
#        if i == rsp_onsets[phase_counter]:
#            current_rsa = rsa["RSA_P2T_Values"][phase_counter]
#            if phase_counter < len(rsp_onsets)-2:
#                phase_counter += 1
#        continuous_rsa.append(current_rsa)
#
#    # Find last phase
#    continuous_rsa = np.array(continuous_rsa)
#    continuous_rsa[max(rsp_onsets):] = np.nan
#
#    df = pd.DataFrame({"RSP":rsp})
#    df["RSA_Values"] = continuous_rsa
#    df["RSA"] = rsa_interpolated
#    rsa["df"] = df
#
#    # Porges–Bohrer method (RSAP–B)
#    # ==============================
#    # Need help to implement this method (See Lewis, 2012)
#
#    return rsa
