# -*- coding: utf-8 -*-
import numpy as np

from ..stats import mad


def hrv_time(heart_period):

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
    bar_y, bar_x = np.histogram(heart_period, bins=range(300, 2000, 8))
    bar_y, bar_x = np.histogram(heart_period, bins="auto")
    out["TINN"] = np.max(bar_x) - np.min(bar_x)  # Triangular Interpolation of the NN Interval Histogram
    out["HTI"] = len(heart_period) / np.max(bar_y)  # HRV Triangular Index

    return out
