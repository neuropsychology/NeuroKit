# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from ..signal.signal_power import signal_power
from ..signal.signal_psd import signal_psd
from .hrv_utils import _hrv_sanitize_input
from .hrv_utils import _hrv_get_rri


def hrv_frequency(peaks, sampling_rate=1000, ulf=(0, 0.0033),
                  vlf=(0.0033, 0.04), lf=(0.04, 0.15), hf=(0.15, 0.4),
                  vhf=(0.4, 0.5), psd_method="welch", show=False, silent=True, **kwargs):
    """ Computes frequency-domain indices of Heart Rate Variability (HRV).

    Note that a minimum duration of the signal containing the peaks is recommended
    for some HRV indices to be meaningful. For instance, 1, 2 and 5 minutes of
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
    psd_method : str
        Method used for spectral density estimation. For details see
        signal.signal_power. By default "welch".
    silent : bool
        If False, warnings will be printed. Default to True.
    show : bool
        If True, will plot the power in the different frequency bands.

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
        - "*LnHF*": the log transformed HF.

    See Also
    --------
    ecg_peaks, ppg_peaks, hrv_summary, hrv_time, hrv_nonlinear

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Download data
    >>> data = nk.data("bio_resting_5min_100hz")
    >>>
    >>> # Find peaks
    >>> peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)
    >>>
    >>> # Compute HRV indices
    >>> hrv = nk.hrv_frequency(peaks, sampling_rate=100, show=True)

    References
    ----------
    - Stein, P. K. (2002). Assessing heart rate variability from real-world
      Holter reports. Cardiac electrophysiology review, 6(3), 239-244.
    - Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate
    variability metrics and norms. Frontiers in public health, 5, 258.
    - `HRV analysis` <https://github.com/Aura-healthcare/hrvanalysis#plot-functions>`_

    """
    # Sanitize input
    peaks = _hrv_sanitize_input(peaks)

    # Compute R-R intervals (also referred to as NN) in milliseconds (interpolated at 1000 Hz by default)
    rri, sampling_rate = _hrv_get_rri(peaks,
                                      sampling_rate=sampling_rate,
                                      interpolate=True, **kwargs)

    power = signal_power(rri,
                         frequency_band=[ulf, vlf, lf, hf, vhf],
                         sampling_rate=sampling_rate,
                         method=psd_method,
                         max_frequency=0.5,
                         **kwargs)

    power.columns = ["ULF", "VLF", "LF", "HF", "VHF"]

    out = power.to_dict(orient="index")[0]

    if silent is False:
        for frequency in out.keys():
            if out[frequency] == 0.0:
                print("Neurokit warning: hrv_frequency(): The duration of recording is too short to allow reliable computation of signal power in frequency band " + frequency + ". Its power is returned as zero.")

    # Normalized
    total_power = np.sum(power.values)
    out["LFHF"] = out["LF"] / out["HF"]
    out["LFn"] = out["LF"] / total_power
    out["HFn"] = out["HF"] / total_power

    # Log
    out["LnHF"] = np.log(out["HF"])

    # Show plot
    if show:
        _hrv_frequency_show(rri, sampling_rate=sampling_rate)

    return out


def _hrv_frequency_show(rri, ulf=(0, 0.0033), vlf=(0.0033, 0.04),
                        lf=(0.04, 0.15), hf=(0.15, 0.4),
                        vhf=(0.4, 0.5), sampling_rate=1000):

    # Get freq psd from rr intervals
    psd = signal_psd(rri, method="welch",
                     sampling_rate=sampling_rate, show=False)
    freq = np.array(psd['Frequency'])
    power = np.array(psd['Power'])

    # Indices between desired frequency bands
    ulf_indexes = np.logical_and(freq >= ulf[0], freq < ulf[1])
    vlf_indexes = np.logical_and(freq >= vlf[0], freq < vlf[1])
    lf_indexes = np.logical_and(freq >= lf[0], freq < lf[1])
    hf_indexes = np.logical_and(freq >= hf[0], freq < hf[1])
    vhf_indexes = np.logical_and(freq >= vhf[0], freq < vhf[1])

    frequency_band_index = [ulf_indexes, vlf_indexes,
                            lf_indexes, hf_indexes, vhf_indexes]
    label_list = ["ULF component", "VLF component",
                  "LF component", "HF component", "VHF component"]

    # Plot
    plt.title("Power Spectral Density (PSD) for Frequency Domains",
              fontsize=10, fontweight="bold")
    plt.xlabel("Frequency (Hz)", fontsize=10)
    plt.ylabel("Power", fontsize=10)

    # Get cmap
    cmap = get_cmap("Set1")
    colors = cmap.colors
    colors = colors[3], colors[1], colors[2], colors[4], colors[0]  # manually rearrange colors

    for band_index, label, i in zip(frequency_band_index, label_list, colors):
        plt.fill_between(freq[band_index], 0, power[band_index],
                         label=label, color=i)
        plt.legend(prop={"size": 10}, loc="best")
        plt.xlim(0, vhf[1])
