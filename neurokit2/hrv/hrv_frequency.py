# -*- coding: utf-8 -*-
import numpy as np
from ..signal.signal_power import signal_power
from ..signal.signal_period import signal_period
from ..signal.signal_interpolate import signal_interpolate


def hrv_frequency(peaks, sampling_rate=1000, sampling_rate_interpolation=10,
                  interpolation_order="cubic", ulf=(0, 0.0033),
                  vlf=(0.0033, 0.04), lf=(0.04, 0.15), hf=(0.15, 0.4),
                  vhf=(0.4, 0.5), method="welch", show=False):
    """[summary]

    Parameters
    ----------
    peaks : [type]
        Samples at which cardiac extrema (R-peaks, systolic peaks) occur.
    sampling_rate : int, optional
        Sampling rate of the continuous cardiac signal in which the peaks occur.
        Should be at least twice as high as the highest frequency in vhf. By
        default 1000.
    sampling_rate_interpolation : int, optional
        Sampling rate at which to interpolate between peaks. By default 10.
    interpolation_order : str, optional
        [description], by default "cubic"
    ulf : tuple, optional
        [description], by default (0, 0.0033)
    vlf : tuple, optional
        [description], by default (0.0033, 0.04)
    lf : tuple, optional
        [description], by default (0.04, 0.15)
    hf : tuple, optional
        [description], by default (0.15, 0.4)
    vhf : tuple, optional
        [description], by default (0.4, 0.5)
    method : str, optional
        [description], by default "welch"
    show : bool, optional
        [description], by default False

    Returns
    -------
    DataFrame
        Contains frequency domain HRV metrics:
        - "*ULF*": spectral power density pertaining to ultra low frequency band i.e., .0 to .0033 Hz by default.
        - "*VLF*": spectral power density pertaining to very low frequency band i.e., .0033 to .04 Hz by default.
        - "*LF*": spectral power density pertaining to low frequency band i.e., .04 to .15 Hz by default.
        - "*HF*": spectral power density pertaining to high frequency band i.e., .15 to .4 Hz by default.
        - "*VHF*": variability, or signal power, in very high frequency i.e., .4 to .5 Hz by default.
        - "*LFHF*": the ratio of low frequency power to high frequency power.
        - "*LFn*": the normalized low frequency, obtained by dividing the low frequency power by the total power.
        - "*HFn*": the normalized high frequency, obtained by dividing the low frequency power by the total power.
    """
    # Compute heart period in milliseconds.
    heart_period = np.diff(peaks) / sampling_rate * 1000

    # Compute length of interpolated heart period signal at requested sampling
    # rate.
    if sampling_rate > sampling_rate_interpolation:
        n_samples = int(np.rint(peaks[-1] / sampling_rate
                                * sampling_rate_interpolation))
    else:
        n_samples = peaks[-1]

    heart_period_intp = signal_interpolate(peaks[1:], heart_period,    # skip first peak since it has no corresponding element in heart_period
                                           desired_length=n_samples,
                                           method=interpolation_order)

    power = signal_power(heart_period_intp,
                         frequency_band=[ulf, vlf, lf, hf, vhf],
                         sampling_rate=sampling_rate_interpolation,
                         method=method, max_frequency=0.5, show=show)
    power.columns = ["ULF", "VLF", "LF", "HF", "VHF"]
    
    out = power.to_dict(orient="index")[0]

    # Normalized
    total_power = np.sum(power.values)
    out["LFHF"] = out["LF"] / out["HF"]
    out["LFn"] = out["LF"] / total_power
    out["HFn"] = out["HF"] / total_power

    # Log
    out["LnHF"] = np.log(out["HF"])

    if show:
        _show(heart_period, out)
    
    return out


def _show(heart_period, out):
    pass
