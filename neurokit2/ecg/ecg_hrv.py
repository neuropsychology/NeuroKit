import pandas as pd
import numpy as np
import scipy.stats

from .ecg_rate import ecg_rate as nk_ecg_rate
from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize
from ..signal import signal_psd




def ecg_hrv(ecg_rate, rpeaks=None, sampling_rate=1000):
    """
    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> ecg = nk.ecg_simulate(duration=120)
    >>> ecg, info = nk.ecg_process(ecg)
    >>> hrv = nk.ecg_hrv(ecg)
    >>> hrv

    References
    ----------
    - Stein, P. K. (2002). Assessing heart rate variability from real-world
      Holter reports. Cardiac electrophysiology review, 6(3), 239-244.
    """
    # Sanitize input
    ecg_rate, rpeaks = _ecg_hrv_formatinput(ecg_rate, rpeaks, sampling_rate)

    # Get raw and interpolated R-R intervals
    rri = np.diff(rpeaks) / sampling_rate * 1000
    ecg_period = ecg_rate / 60 * sampling_rate

    # Get indices
    hrv = {}  # Initialize empty dict
    hrv.update(_ecg_hrv_time(rri))

    hrv = pd.DataFrame.from_dict(hrv, orient='index').T
    return hrv


# =============================================================================
# Methods (Domains)
# =============================================================================



def _ecg_hrv_time(rri):
    out = {}  # Initialize empty dict

    # Mean based
    out["RMSSD"] = np.sqrt(np.mean(np.diff(rri) ** 2))
    out["MeanNN"] = np.mean(rri)
    out["SDNN"] = np.std(rri, ddof=1)
    out["SDSD"] = np.std(np.diff(rri), ddof=1)
    out["CVNN"] = out["SDNN"] / out["MeanNN"]
    out["CVSD"] = out["RMSSD"] / out["MeanNN"]

    # Robust
    out["MedianNN"] = np.median(np.abs(rri))
    out["MadNN"] = scipy.stats.median_absolute_deviation(rri)
    out["MCVNN"] = out["MadNN"] / out["MedianNN"]

    # Extreme-based
    nn50 = np.sum(np.abs(np.diff(rri)) > 50)
    nn20 = np.sum(np.abs(np.diff(rri)) > 20)
    out["pNN50"] = nn50 / len(rri) * 100
    out["pNN20"] = nn20 / len(rri) * 100

    # Histogram-based
    bar_y, bar_x = np.histogram(rri, bins="auto")
    out["TINN"] = np.max(bar_x) - np.min(bar_x) # Triangular Interpolation of the NN Interval Histogram
    out["HTI"] = len(rri) / np.max(bar_y) # HRV Triangular Index

    return out


#def _ecg_hrv_frequency(ecg_period, ulf=[0, 0.0033], vlf=[0.0033, 0.04], lf=[0.04, 0.15], hf=[0.15, 0.4], vhf=[0.4, 0.5], method="welch"):
#
#    psd = signal_psd(ecg_period, sampling_rate=1000, method=method, max_frequency=0.5, show=False)
#
#    ulf_indices = np.logical_and(psd["Frequency"] >= ulf[0], psd["Frequency"] < ulf[1]).values
#    vlf_indices = np.logical_and(psd["Frequency"] >= vlf[0], psd["Frequency"] < vlf[1]).values
#    lf_indices = np.logical_and(psd["Frequency"] >= lf[0], psd["Frequency"] < lf[1]).values
#    hf_indices = np.logical_and(psd["Frequency"] >= hf[0], psd["Frequency"] < hf[1]).values
#
#    out = {}  # Initialize empty dict
#    out["ULF"] = np.trapz(y=psd["Power"][ulf_indices], x=psd["Frequency"][ulf_indices])
#    vlf = np.trapz(y=psd["Power"][vlf_indices], x=psd["Frequency"][vlf_indices])
#    lf = np.trapz(y=psd["Power"][lf_indices], x=psd["Frequency"][lf_indices])
#    hf = np.trapz(y=psd["Power"][hf_indices], x=psd["Frequency"][hf_indices])
#
#
#    total_power = ulf + vlf + lf + hf
#    lf_hf = lf / hf
#    lfnu = (lf / (total_power - vlf)) * 100
#    hfnu = (hf / (total_power - vlf)) * 100



# =============================================================================
# Internals
# =============================================================================

def _ecg_hrv_formatinput(ecg_rate, rpeaks=None, sampling_rate=1000):

    if isinstance(ecg_rate, pd.DataFrame):
        df = ecg_rate.copy()
        cols = [col for col in df.columns if 'ECG_Rate' in col]
        if len(cols) == 0:
            cols = [col for col in df.columns if 'ECG_R_Peaks' in col]
            if len(cols) == 0:
                raise ValueError("NeuroKit error: _ecg_hrv_formatinput(): Wrong input, ",
                                 "we couldn't extract ecg_rate and rpeaks indices.")
            else:
                ecg_rate = nk_ecg_rate(rpeaks, sampling_rate=sampling_rate, desired_length=len(df))
        else:
            ecg_rate = df[cols[0]].values



    if rpeaks is None:
        try:
            rpeaks, _ = _signal_formatpeaks_sanitize(df, desired_length=None)
        except NameError:
            raise ValueError("NeuroKit error: _ecg_hrv_formatinput(): Wrong input, ",
                             "we couldn't extract rpeaks indices.")
    else:
        rpeaks, _ = _signal_formatpeaks_sanitize(rpeaks, desired_length=None)

    return ecg_rate, rpeaks
