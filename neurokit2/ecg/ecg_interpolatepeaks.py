from math import cos, pi
import numpy as np
import pandas as pd


def ecg_interpolatepeaks(ecg_peaks):
    """**Interpolate R-peaks in an ECG signal**

    Trigonometrically interpolates between R-peaks. This approach considers the intrinsic 
    periodical property of heartbeats, instead of a mere linear interpolation between IBIs.
    As a period can only be calculated with a known start and ending, the entries before the
    first R-peak and the entries after the last R-peak will be np.NaNs. The length of the 
    original time series is preserved.

    Parameters
    ----------
    ecg_peaks : tuple
        Takes the output of ecg_peaks().

    Returns
    ----------
    rpeaks_cip : pd.Series
        A cosine interpolation of the R-peaks between the values 1 and 0 for y, where 1 stands
        for an R-peak and 0 stands for the center point between two subsequent R-peaks.
    """
    def cosine(x, x1, x2):
        # Returns cosine value (y) [0,1] for given time step (x) in one period (T) based on
        # the interval between x1 and x2.
        # Args:
        #    x (float): time step
        #    x1 (float): period start (y=1)
        #    x2 (float): period end (y=1)
        # Returns:
        #    val (float): cosine value [0,1]

        T = x2-x1
        val = 0.5+0.5*cos(2.0 * pi * (x - x1) / T)
        return val

    length = len(ecg_peaks[0])
    rpeaks_idx = ecg_peaks[1]["ECG_R_Peaks"]
    num_rpeaks = len(rpeaks_idx)

    # Interpolation
    rpeaks_cip = [np.nan]*(rpeaks_idx[0])
    for i in range(num_rpeaks):
        if i == num_rpeaks-1:
            break
        x1 = rpeaks_idx[i]
        x2 = rpeaks_idx[i+1]
        T = x2-x1
        def f(x): return cosine(x, x1, x2)
        # span interval with corresponding number of time steps
        x = np.linspace(x1, x2, T)
        y = [f(x) for x in x]
        rpeaks_cip = rpeaks_cip + y
    rpeaks_cip = rpeaks_cip + [np.nan]*(length-rpeaks_idx[-1])
    # keep in mind: there are nans at the start and end

    return pd.Series(rpeaks_cip)
