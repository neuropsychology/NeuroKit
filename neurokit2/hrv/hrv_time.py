# -*- coding: utf-8 -*-
import numpy as np
from ..stats.mad import mad


def hrv_time(peaks, sampling_rate=1000, show=False):
    """[summary]

    Parameters
    ----------
    peaks : [type]
        Samples at which cardiac extrema (R-peaks, systolic peaks) occur.
    sampling_rate : int, optional
        Sampling rate of the continuous cardiac signal in which the peaks occur.
        Should be at least twice as high as the highest frequency in vhf. By
        default 1000.
    show : bool, optional
        
    Returns
    -------
    DataFrame
        Contains time domain HRV metrics:
        - "*RMSSD*": the square root of the mean of the sum of successive differences between adjacent RR intervals.
        - "*MeanNN*": the mean of the RR intervals.
        - "*SDNN*": the standard deviation of the RR intervals.
        - "*SDSD*": the standard deviation of the successive differences between RR intervals.
        - "*CVNN*": the standard deviation of the RR intervals (SDNN) divided by the mean of the RR intervals (MeanNN).
        - "*CVSD*": the root mean square of the sum of successive differences (RMSSD) divided by the mean of the RR intervals (MeanNN).
        - "*MedianNN*": the median of the absolute values of the successive differences between RR intervals.
        - "*MadNN*": the median absolute deviation of the RR intervals.
        - "*HCVNN*": the median absolute deviation of the RR intervals (MadNN) divided by the median of the absolute differences of their successive differences (MedianNN).
        - "*pNN50*": the proportion of RR intervals greater than 50ms, out of the total number of RR intervals.
        - "*pNN20*": the proportion of RR intervals greater than 20ms, out of the total number of RR intervals.
        - "*TINN*": a geometrical parameter of the HRV, or more specifically, the baseline width of the RR intervals distribution obtained by triangular interpolation, where the error of least squares determines the triangle. It is an approximation of the RR interval distribution.
        - "*HTI*": the HRV triangular index, measuring the total number of RR intervals divded by the height of the RR intervals histogram.
    """

    # Compute heart period in milliseconds.
    heart_period = np.diff(peaks) / sampling_rate * 1000
    
    diff_period = np.diff(heart_period)
    
    out = {}  # Initialize empty dict

    # Mean based
    out["RMSSD"] = np.sqrt(np.mean(diff_period ** 2))
    out["MeanNN"] = np.mean(heart_period)
    out["SDNN"] = np.std(heart_period, ddof=1)
    out["SDSD"] = np.std(diff_period, ddof=1)
    out["CVNN"] = out["SDNN"] / out["MeanNN"]
    out["CVSD"] = out["RMSSD"] / out["MeanNN"]

    # Robust
    out["MedianNN"] = np.median(np.abs(heart_period))
    out["MadNN"] = mad(heart_period)
    out["MCVNN"] = out["MadNN"] / out["MedianNN"]

    # Extreme-based
    nn50 = np.sum(np.abs(diff_period) > 50)
    nn20 = np.sum(np.abs(diff_period) > 20)
    out["pNN50"] = nn50 / len(heart_period) * 100
    out["pNN20"] = nn20 / len(heart_period) * 100

    # Geometrical domain
    bar_y, bar_x = np.histogram(heart_period, bins="auto")
    out["TINN"] = np.max(bar_x) - np.min(bar_x)  # Triangular Interpolation of the NN Interval Histogram
    out["HTI"] = len(heart_period) / np.max(bar_y)  # HRV Triangular Index

    if show:
        _show(heart_period, out)
    
    return out


def _show(heart_period, out):
    pass
