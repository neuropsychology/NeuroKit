import pandas as pd
import numpy as np


#def ecg_hrv(ecg_rate, sampling_rate=1000):
#    """
#    Examples
#    --------
#    >>> import neurokit2 as nk
#    >>> ecg = nk.ecg_simulate(duration=60)
#    >>> ecg, info = nk.ecg_process(ecg)
#    >>> ecg_rate = ecg["ECG_Rate"]
#    """
#
#    rri = ecg_rate / 60 * sampling_rate





#def _ecg_hrv_timedomain(RRis, sampling_rate=1000):
#    out = {}
#    out["RMSSD"] = np.sqrt(np.mean(np.diff(RRis) ** 2))
#    out["meanNN"] = np.mean(RRis)
#    out["sdNN"] = np.std(RRis, ddof=1)  # make it calculate N-1
#    out["cvNN"] = out["sdNN"] / out["meanNN"]
#    out["CVSD"] = out["RMSSD"] / out["meanNN"]
#    out["medianNN"] = np.median(abs(RRis))
#    out["madNN"] = mad(RRis, constant=1)
#    out["mcvNN"] = out["madNN"] / out["medianNN"]
#    nn50 = sum(abs(np.diff(RRis)) > 50)
#    nn20 = sum(abs(np.diff(RRis)) > 20)
#    out["pNN50"] = nn50 / len(RRis) * 100
#    out["pNN20"] = nn20 / len(RRis) * 100