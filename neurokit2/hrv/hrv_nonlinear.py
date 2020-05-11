# -*- coding: utf-8 -*-
import numpy as np

from ..complexity import entropy_sample


def _ecg_hrv_nonlinear(heart_period, heart_period_intp):
    diff_heart_period = np.diff(heart_period)
    out = {}

    # Poincaré plot
    sd_heart_period = np.std(diff_heart_period, ddof=1) ** 2
    out["SD1"] = np.sqrt(sd_heart_period * 0.5)
    out["SD2"] = np.sqrt(2 * sd_heart_period - 0.5 * sd_heart_period)
    out["SD2SD1"] = out["SD2"] / out["SD1"]

    # CSI / CVI
    T = 4 * out["SD1"]
    L = 4 * out["SD2"]
    out["CSI"] = L / T
    out["CVI"] = np.log10(L * T)
    out["CSI_Modified"] = L ** 2 / T

    # Entropy
    out["SampEn"] = entropy_sample(heart_period, dimension=2,
                                   r=0.2 * np.std(heart_period, ddof=1))

    return out
