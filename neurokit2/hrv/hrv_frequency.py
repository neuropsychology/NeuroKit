# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from ..signal.signal_power import _signal_power_instant_plot, signal_power
from ..signal.signal_psd import signal_psd
from .hrv_utils import _hrv_get_rri, _hrv_sanitize_input


def hrv_frequency(
    peaks,
    sampling_rate=1000,
    ulf=(0, 0.0033),
    vlf=(0.0033, 0.04),
    lf=(0.04, 0.15),
    hf=(0.15, 0.4),
    vhf=(0.4, 0.5),
    psd_method="welch",
    show=False,
    silent=True,
    normalize=True,
    order_criteria=None,
    **kwargs
):
    """Computes frequency-domain indices of Heart Rate Variability (HRV).

    Note that a minimum duration of the signal containing the peaks is recommended for some HRV indices
    to be meaningful. For instance, 1, 2 and 5 minutes of high quality signal are the recomended
    minima for HF, LF and LF/HF, respectively. See references for details.

    Parameters
    ----------
    peaks : dict
        Samples at which cardiac extrema (i.e., R-peaks, systolic peaks) occur. Dictionary returned
        by ecg_findpeaks, ecg_peaks, ppg_findpeaks, or ppg_peaks.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks occur. Should be at
        least twice as high as the highest frequency in vhf. By default 1000.
    ulf : tuple, optional
        Upper and lower limit of the ultra-low frequency band. By default (0, 0.0033).
    vlf : tuple, optional
        Upper and lower limit of the very-low frequency band. By default (0.0033, 0.04).
    lf : tuple, optional
        Upper and lower limit of the low frequency band. By default (0.04, 0.15).
    hf : tuple, optional
        Upper and lower limit of the high frequency band. By default (0.15, 0.4).
    vhf : tuple, optional
        Upper and lower limit of the very-high frequency band. By default (0.4, 0.5).
    psd_method : str
        Method used for spectral density estimation. For details see signal.signal_power. By default "welch".
    silent : bool
        If False, warnings will be printed. Default to True.
    show : bool
        If True, will plot the power in the different frequency bands.
    normalize : bool
        Normalization of power by maximum PSD value. Default to True.
        Normalization allows comparison between different PSD methods.
    order_criteria : str
        The criteria to automatically select order in parametric PSD (only used for autoregressive
        (AR) methods such as 'burg'). Defaults to None.
    **kwargs : optional
        Other arguments.

    Returns
    -------
    DataFrame
        Contains frequency domain HRV metrics:
        - **ULF**: The spectral power density pertaining to ultra low frequency band i.e., .0 to .0033 Hz
        by default.
        - **VLF**: The spectral power density pertaining to very low frequency band i.e., .0033 to .04 Hz
        by default.
        - **LF**: The spectral power density pertaining to low frequency band i.e., .04 to .15 Hz by default.
        - **HF**: The spectral power density pertaining to high frequency band i.e., .15 to .4 Hz by default.
        - **VHF**: The variability, or signal power, in very high frequency i.e., .4 to .5 Hz by default.
        - **LFn**: The normalized low frequency, obtained by dividing the low frequency power by
        the total power.
        - **HFn**: The normalized high frequency, obtained by dividing the low frequency power by
        the total power.
        - **LnHF**: The log transformed HF.

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
    >>> hrv_welch = nk.hrv_frequency(peaks, sampling_rate=100, show=True, psd_method="welch")
    >>> hrv_burg = nk.hrv_frequency(peaks, sampling_rate=100, show=True, psd_method="burg")
    >>> hrv_lomb = nk.hrv_frequency(peaks, sampling_rate=100, show=True, psd_method="lomb")
    >>> hrv_multitapers = nk.hrv_frequency(peaks, sampling_rate=100, show=True, psd_method="multitapers")

    References
    ----------
    - Stein, P. K. (2002). Assessing heart rate variability from real-world Holter reports. Cardiac
    electrophysiology review, 6(3), 239-244.

    - Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms.
    Frontiers in public health, 5, 258.

    - Boardman, A., Schlindwein, F. S., & Rocha, A. P. (2002). A study on the optimum order of
    autoregressive models for heart rate variability. Physiological measurement, 23(2), 325.

    - Bachler, M. (2017). Spectral Analysis of Unevenly Spaced Data: Models and Application in Heart
    Rate Variability. Simul. Notes Eur., 27(4), 183-190.

    """
    # Sanitize input
    peaks = _hrv_sanitize_input(peaks)

    # Compute R-R intervals (also referred to as NN) in milliseconds (interpolated at 1000 Hz by default)
    rri, sampling_rate = _hrv_get_rri(peaks, sampling_rate=sampling_rate, interpolate=True, **kwargs)

    frequency_band = [ulf, vlf, lf, hf, vhf]
    power = signal_power(
        rri,
        frequency_band=frequency_band,
        sampling_rate=sampling_rate,
        method=psd_method,
        max_frequency=0.5,
        show=False,
        normalize=normalize,
        order_criteria=order_criteria,
        **kwargs
    )

    power.columns = ["ULF", "VLF", "LF", "HF", "VHF"]

    out = power.to_dict(orient="index")[0]
    out_bands = out.copy()  # Components to be entered into plot

    if silent is False:
        for frequency in out.keys():
            if out[frequency] == 0.0:
                warn(
                    "The duration of recording is too short to allow"
                    " reliable computation of signal power in frequency band " + frequency + "."
                    " Its power is returned as zero.",
                    category=NeuroKitWarning
                )

    # Normalized
    total_power = np.nansum(power.values)
    out["LFHF"] = out["LF"] / out["HF"]
    out["LFn"] = out["LF"] / total_power
    out["HFn"] = out["HF"] / total_power

    # Log
    out["LnHF"] = np.log(out["HF"])  # pylint: disable=E1111

    out = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("HRV_")

    # Plot
    if show:
        _hrv_frequency_show(rri, out_bands, sampling_rate=sampling_rate, psd_method=psd_method, order_criteria=order_criteria, normalize=normalize)
    return out


def _hrv_frequency_show(
    rri,
    out_bands,
    ulf=(0, 0.0033),
    vlf=(0.0033, 0.04),
    lf=(0.04, 0.15),
    hf=(0.15, 0.4),
    vhf=(0.4, 0.5),
    sampling_rate=1000,
    psd_method="welch",
    order_criteria=None,
    normalize=True,
    **kwargs
):

    if "ax" in kwargs:
        ax = kwargs.get("ax")
        kwargs.pop("ax")
    else:
        __, ax = plt.subplots()

    frequency_band = [ulf, vlf, lf, hf, vhf]
    for i in range(len(frequency_band)):  # pylint: disable=C0200
        min_frequency = frequency_band[i][0]
        if min_frequency == 0:
            min_frequency = 0.001  # sanitize lowest frequency

        window_length = int((2 / min_frequency) * sampling_rate)
        if window_length <= len(rri) / 2:
            break

    psd = signal_psd(rri, sampling_rate=sampling_rate, show=False, min_frequency=min_frequency, method=psd_method, max_frequency=0.5, order_criteria=order_criteria, normalize=normalize)

    _signal_power_instant_plot(psd, out_bands, frequency_band, ax=ax)
