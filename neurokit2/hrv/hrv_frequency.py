# -*- coding: utf-8 -*-
import numpy as np
from ..signal.signal_power import signal_power
from ..signal.signal_period import signal_period
from ..signal.signal_interpolate import signal_interpolate


def hrv_frequency(peaks, sampling_rate=1000, sampling_rate_interpolation=10,
                  interpolation_order="cubic", ulf=(0, 0.0033),
                  vlf=(0.0033, 0.04), lf=(0.04, 0.15), hf=(0.15, 0.4),
                  vhf=(0.4, 0.5), psd_method="welch"):
    """ Computes frequency-domain indices of Heart Rate Variability (HRV).
    
    Note that a minimum duration of the signal containing the peaks is recommended
    for some HRV indices to be meaninful. For instance, 1, 2 and 5 minutes of
    high quality signal are the recomended minima for HF, LF and LF/HF,
    respectively. See references for details.

    Parameters
    ----------
    peaks : dict
        Samples at which cardiac extrema (i.e., R-peaks, systolic peaks) occur.
        Dictionary returned by ecg_findpeaks, ecg_peaks, ppg_findpeaks, or
        ppg_peaks.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks
        occur. Should be at least twice as high as the highest frequency in vhf.
        By default 1000.
    sampling_rate_interpolation : int, optional
        Sampling rate (Hz) at which to interpolate between peaks. By default 10.
    interpolation_order : str, optional
        Method used to interpolate heart period. For details see
        signal.signal_interpolate. By default "cubic".
    ulf : tuple, optional
        Upper and lower limit of the ultra-low frequency band. By default
        (0, 0.0033).
    vlf : tuple, optional
        Upper and lower limit of the very-low frequency band. By default
        (0.0033, 0.04).
    lf : tuple, optional
        Upper and lower limit of the low frequency band. By default (0.04, 0.15).
    hf : tuple, optional
        Upper and lower limit of the high frequency band. By default (0.15, 0.4).
    vhf : tuple, optional
        Upper and lower limit of the very-high frequency band. By default
        (0.4, 0.5).
    psd_method : str, optional
        Method used for spectral density estimation. For details see
        signal.signal_power. By default "welch".

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
    
    See Also
    --------
    ecg_peaks, ppg_peaks, hrv_summary, hrv_time, hrv_nonlinear

    Examples
    --------
    
    References
    ----------
    - Stein, P. K. (2002). Assessing heart rate variability from real-world
      Holter reports. Cardiac electrophysiology review, 6(3), 239-244.
    - Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate
    variability metrics and norms. Frontiers in public health, 5, 258.
    """
    error = ValueError("NeuroKit error: Please pass in a valid dictionary"
                       " as the peaks argument (see docstring).")
    if not isinstance(peaks, dict):
        raise error
    if len(peaks.keys()) != 1:
        raise error
    if [*peaks.keys()][0] not in ["ECG_R_Peaks", "PPG_Peaks"]:
        raise error
    
    peaks = [*peaks.values()][0]
    
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
                         method=psd_method, max_frequency=0.5)
    power.columns = ["ULF", "VLF", "LF", "HF", "VHF"]
    
    out = power.to_dict(orient="index")[0]

    # Normalized
    total_power = np.sum(power.values)
    out["LFHF"] = out["LF"] / out["HF"]
    out["LFn"] = out["LF"] / total_power
    out["HFn"] = out["HF"] / total_power

    # Log
    out["LnHF"] = np.log(out["HF"])

    # if show:
    #     _show(heart_period, out)
    
    return out


def _show(heart_period, out):
    pass
