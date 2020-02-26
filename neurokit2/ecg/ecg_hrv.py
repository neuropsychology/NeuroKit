import pandas as pd
import numpy as np
import scipy.stats

from .ecg_rate import ecg_rate as nk_ecg_rate
from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize





def ecg_hrv(ecg_rate, rpeaks=None, sampling_rate=1000):
    """
    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> ecg = nk.ecg_simulate(duration=60)
    >>> ecg, info = nk.ecg_process(ecg)
    >>> hrv = nk.ecg_hrv(ecg)
    >>> hrv
    """
    # Sanitize input
    ecg_rate, rpeaks = _ecg_hrv_formatinput(ecg_rate, rpeaks, sampling_rate)

    # Get raw and interpolated R-R intervals
    rri = np.diff(rpeaks) / sampling_rate
    ecg_period = ecg_rate / 60 * sampling_rate

    timedomain = _ecg_hrv_timedomain(rri)

    return timedomain






def _ecg_hrv_timedomain(rri):
    out = {}  # Initialize empty dict

    # Mean based
    out["RMSSD"] = np.sqrt(np.mean(np.diff(rri) ** 2))
    out["meanNN"] = np.mean(rri)
    out["sdNN"] = np.std(rri, ddof=1)  # make it calculate N-1
    out["cvNN"] = out["sdNN"] / out["meanNN"]
    out["CVSD"] = out["RMSSD"] / out["meanNN"]

    # Robust
    out["medianNN"] = np.median(np.abs(rri))
    out["madNN"] = scipy.stats.median_absolute_deviation(rri)
    out["mcvNN"] = out["madNN"] / out["medianNN"]

    # Extreme-based
    nn50 = np.sum(np.abs(np.diff(rri)) > 50)
    nn20 = np.sum(np.abs(np.diff(rri)) > 20)
    out["pNN50"] = nn50 / len(rri) * 100
    out["pNN20"] = nn20 / len(rri) * 100
    return out



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
