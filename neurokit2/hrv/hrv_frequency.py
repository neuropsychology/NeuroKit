# -*- coding: utf-8 -*-
import numpy as np

from ..signal import signal_power


def hrv_frequency(heart_period, sampling_rate=1000, ulf=(0, 0.0033),
                  vlf=(0.0033, 0.04), lf=(0.04, 0.15), hf=(0.15, 0.4),
                  vhf=(0.4, 0.5), method="welch", show=False):

    power = signal_power(heart_period, frequency_band=[ulf, vlf, lf, hf, vhf],
                         sampling_rate=sampling_rate, method=method,
                         max_frequency=0.5, show=show)
    power.columns = ["ULF", "VLF", "LF", "HF", "VHF"]
    out = power.to_dict(orient="index")[0]

    # Normalized
    total_power = np.sum(power.values)
    out["LFHF"] = out["LF"] / out["HF"]
    out["LFn"] = out["LF"] / total_power
    out["HFn"] = out["HF"] / total_power

    # Log
    out["LnHF"] = np.log(out["HF"])

    return out
