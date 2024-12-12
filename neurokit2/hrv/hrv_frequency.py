# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from ..signal.signal_power import _signal_power_instant_plot, signal_power
from ..signal.signal_psd import signal_psd
from .hrv_utils import _hrv_format_input
from .intervals_process import intervals_process


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
    interpolation_rate=100,
    **kwargs
):
    """**Computes frequency-domain indices of Heart Rate Variability (HRV)**

    Computes frequency domain HRV metrics, such as the power in different frequency bands.

    * **ULF**: The spectral power of ultra low frequencies (by default, .0 to
      .0033 Hz). Very long signals are required for this to index to be
      extracted, otherwise, will return NaN.
    * **VLF**: The spectral power of very low frequencies (by default, .0033 to .04 Hz).
    * **LF**: The spectral power of low frequencies (by default, .04 to .15 Hz).
    * **HF**: The spectral power of high frequencies (by default, .15 to .4 Hz).
    * **VHF**: The spectral power of very high frequencies (by default, .4 to .5 Hz).
    * **TP**: The total spectral power.
    * **LFHF**: The ratio obtained by dividing the low frequency power by the high frequency power.
    * **LFn**: The normalized low frequency, obtained by dividing the low frequency power by
      the total power.
    * **HFn**: The normalized high frequency, obtained by dividing the low frequency power by
      the total power.
    * **LnHF**: The log transformed HF.

    Note that a minimum duration of the signal containing the peaks is recommended for some HRV
    indices to be meaningful. For instance, 1, 2 and 5 minutes of high quality signal are the
    recommended minima for HF, LF and LF/HF, respectively.

    .. tip::

      We strongly recommend checking our open-access paper `Pham et al. (2021)
      <https://doi.org/10.3390/s21123998>`_ on HRV indices for more information.


    Parameters
    ----------
    peaks : dict
        Samples at which cardiac extrema (i.e., R-peaks, systolic peaks) occur.
        Can be a list of indices or the output(s) of other functions such as :func:`.ecg_peaks`,
        :func:`.ppg_peaks`, :func:`.ecg_process` or :func:`.bio_process`.
        Can also be a dict containing the keys `RRI` and `RRI_Time`
        to directly pass the R-R intervals and their timestamps, respectively.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks occur.
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
        Method used for spectral density estimation. For details see :func:`.signal_power`.
        By default ``"welch"``.
    silent : bool
        If ``False``, warnings will be printed. Default to ``True``.
    show : bool
        If ``True``, will plot the power in the different frequency bands.
    normalize : bool
        Normalization of power by maximum PSD value. Default to ``True``.
        Normalization allows comparison between different PSD methods.
    order_criteria : str
        The criteria to automatically select order in parametric PSD (only used for autoregressive
        (AR) methods such as ``"burg"``). Defaults to ``None``.
    interpolation_rate : int, optional
        Sampling rate (Hz) of the interpolated interbeat intervals. Should be at least twice as
        high as the highest frequency in vhf. By default 100. To replicate Kubios defaults, set to 4.
        To not interpolate, set interpolation_rate to None (in case the interbeat intervals are already
        interpolated or when using the ``"lombscargle"`` psd_method for which interpolation is not required).
    **kwargs
        Additional other arguments.

    Returns
    -------
    DataFrame
        DataFrame consisting of the computed HRV frequency metrics, which includes:

        .. codebookadd::
            HRV_ULF|The spectral power of ultra low frequencies (by default, .0 to .0033 Hz). \
                Very long signals are required for this to index to be extracted, otherwise, \
                will return NaN.
            HRV_VLF|The spectral power of very low frequencies (by default, .0033 to .04 Hz).
            HRV_LF|The spectral power of low frequencies (by default, .04 to .15 Hz).
            HRV_HF|The spectral power of high frequencies (by default, .15 to .4 Hz).
            HRV_VHF|The spectral power of very high frequencies (by default, .4 to .5 Hz).
            HRV_TP|The total spectral power.
            HRV_LFHF|The ratio obtained by dividing the low frequency power by the high frequency \
                power.
            HRV_LFn|The normalized low frequency, obtained by dividing the low frequency power by \
                the total power.
            HRV_HFn|The normalized high frequency, obtained by dividing the low frequency power by \
                the total power.
            HRV_LnHF|The log transformed HF.

    See Also
    --------
    ecg_peaks, ppg_peaks, hrv_summary, hrv_time, hrv_nonlinear

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Download data
      data = nk.data("bio_resting_5min_100hz")

      # Find peaks
      peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)

      # Compute HRV indices using method="welch"
      @savefig p_hrv_freq1.png scale=100%
      hrv_welch = nk.hrv_frequency(peaks, sampling_rate=100, show=True, psd_method="welch")
      @suppress
      plt.close()

    .. ipython:: python

      # Using method ="burg"
      @savefig p_hrv_freq2.png scale=100%
      hrv_burg = nk.hrv_frequency(peaks, sampling_rate=100, show=True, psd_method="burg")
      @suppress
      plt.close()

    .. ipython:: python

      # Using method = "lomb" (requires installation of astropy)
      @savefig p_hrv_freq3.png scale=100%
      hrv_lomb = nk.hrv_frequency(peaks, sampling_rate=100, show=True, psd_method="lomb")
      @suppress
      plt.close()

    .. ipython:: python

      # Using method="multitapers"
      @savefig p_hrv_freq4.png scale=100%
      hrv_multitapers = nk.hrv_frequency(peaks, sampling_rate=100, show=True,psd_method="multitapers")
      @suppress
      plt.close()

    References
    ----------
    * Pham, T., Lau, Z. J., Chen, S. H. A., & Makowski, D. (2021). Heart Rate Variability in
      Psychology: A Review of HRV Indices and an Analysis Tutorial. Sensors, 21(12), 3998.
    * Stein, P. K. (2002). Assessing heart rate variability from real-world Holter reports. Cardiac
      electrophysiology review, 6(3), 239-244.
    * Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and
      norms. Frontiers in public health, 5, 258.
    * Boardman, A., Schlindwein, F. S., & Rocha, A. P. (2002). A study on the optimum order of
      autoregressive models for heart rate variability. Physiological measurement, 23(2), 325.
    * Bachler, M. (2017). Spectral Analysis of Unevenly Spaced Data: Models and Application in Heart
      Rate Variability. Simul. Notes Eur., 27(4), 183-190.

    """

    # Sanitize input
    # If given peaks, compute R-R intervals (also referred to as NN) in milliseconds
    rri, rri_time, _ = _hrv_format_input(peaks, sampling_rate=sampling_rate)

    # Process R-R intervals (interpolated at 100 Hz by default)
    rri, rri_time, sampling_rate = intervals_process(
        rri, intervals_time=rri_time, interpolate=True, interpolation_rate=interpolation_rate, **kwargs
    )

    if interpolation_rate is None:
        t = rri_time
    else:
        t = None

    frequency_band = [ulf, vlf, lf, hf, vhf]

    # Find maximum frequency
    max_frequency = np.max([np.max(i) for i in frequency_band])

    power = signal_power(
        rri,
        frequency_band=frequency_band,
        sampling_rate=sampling_rate,
        method=psd_method,
        max_frequency=max_frequency,
        show=False,
        normalize=normalize,
        order_criteria=order_criteria,
        t=t,
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
                    category=NeuroKitWarning,
                )

    # Normalized
    total_power = np.nansum(power.values)
    out["TP"] = total_power
    out["LFHF"] = out["LF"] / out["HF"]
    out["LFn"] = out["LF"] / total_power
    out["HFn"] = out["HF"] / total_power

    # Log
    out["LnHF"] = np.log(out["HF"])  # pylint: disable=E1111

    out = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("HRV_")

    # Plot
    if show:
        _hrv_frequency_show(
            rri,
            out_bands,
            ulf=ulf,
            vlf=vlf,
            lf=lf,
            hf=hf,
            vhf=vhf,
            sampling_rate=sampling_rate,
            psd_method=psd_method,
            order_criteria=order_criteria,
            normalize=normalize,
            max_frequency=max_frequency,
            t=t,
        )
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
    max_frequency=0.5,
    t=None,
    **kwargs
):

    if "ax" in kwargs:
        ax = kwargs.get("ax")
        kwargs.pop("ax")
    else:
        __, ax = plt.subplots()

    frequency_band = [ulf, vlf, lf, hf, vhf]

    # Compute sampling rate for plot windows
    if sampling_rate is None:
        med_sampling_rate = np.median(np.diff(t))  # This is just for visualization purposes (#800)
    else:
        med_sampling_rate = sampling_rate

    for i in range(len(frequency_band)):  # pylint: disable=C0200
        min_frequency = frequency_band[i][0]
        if min_frequency == 0:
            min_frequency = 0.001  # sanitize lowest frequency

        window_length = int((2 / min_frequency) * med_sampling_rate)
        if window_length <= len(rri) / 2:
            break

    psd = signal_psd(
        rri,
        sampling_rate=sampling_rate,
        show=False,
        min_frequency=min_frequency,
        method=psd_method,
        max_frequency=max_frequency,
        order_criteria=order_criteria,
        normalize=normalize,
        t=t,
    )

    _signal_power_instant_plot(psd, out_bands, frequency_band, ax=ax)
