# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal import signal_filter, signal_interpolate, signal_rate, signal_resample
from .rsp_peaks import rsp_peaks


def rsp_rate(
    rsp_cleaned,
    troughs=None,
    sampling_rate=1000,
    window=10,
    hop_size=1,
    method="trough",
    peak_method="khodadad2018",
    interpolation_method="monotone_cubic",
):
    """**Find respiration rate**

    Parameters
    ----------
    rsp_cleaned : Union[list, np.array, pd.Series]
        The cleaned respiration channel as returned by :func:`.rsp_clean`.
    troughs : Union[list, dict, np.array, pd.Series, pd.DataFrame]
        The respiration troughs (inhalation onsets) as returned by :func:`.rsp_peaks`.
        If None (default), inhalation onsets will be automatically identified from the
        :func:`.rsp_clean` signal.
    sampling_rate : int
        The sampling frequency of :func:`.rsp_cleaned` (in Hz, i.e., samples/second).
    window : int
        The duration of the sliding window (in second). Default to 10 seconds.
    hop_size : int
        The number of samples between each successive window. Default to 1 sample.
    method : str
        Method can either be ``"trough"`` or ``"xcorr"``. In ``"trough"`` method, respiratory rate
        is calculated from the periods between successive inspirations (i.e., inhalation onsets/
        troughs). In ``"xcorr"`` method, cross-correlations between the changes in respiration with
        a bank of sinusoids of different frequencies are calculated to identify the principal
        frequency of oscillation.
    peak_method : str
        Method to identify successive respiratory inspirations, only relevant if method is
        ``"trough"``. Can be one of ``"khodadad2018"`` (default) or ``"biosppy"``.
    interpolation_method : str
        Method used to interpolate the rate between inhalation onsets.
        See :func:`.signal_interpolate`. ``"monotone_cubic"`` is chosen as the default
        interpolation method since it ensures monotone interpolation between data points (i.e., it
        prevents physiologically implausible "overshoots" or "undershoots" in the
        y-direction). In contrast, the widely used cubic spline interpolation does not ensure
        monotonicity.

    Return
    ------
    rsp_rate : np.ndarray
        Instantenous respiration rate.

    Example
    -------
    .. ipython:: python

      import neurokit2 as nk
      rsp_signal = nk.data("rsp_1000hz")

      rsp_cleaned = nk.rsp_clean(rsp_signal, sampling_rate=1000)
      rsp_rate_onsets = nk.rsp_rate(rsp_cleaned, sampling_rate=1000, method="trough")
      rsp_rate_xcorr = nk.rsp_rate(rsp_cleaned, sampling_rate=1000, method="xcorr")

    """

    if method.lower() in ["period", "peak", "peaks", "trough", "troughs", "signal_rate"]:
        if troughs is None:
            _, troughs = rsp_peaks(rsp_cleaned, sampling_rate=sampling_rate, method=peak_method)
        if isinstance(troughs, (pd.DataFrame, dict)):
            troughs = troughs["RSP_Troughs"]
        rate = signal_rate(
            troughs,
            sampling_rate=sampling_rate,
            desired_length=len(rsp_cleaned),
            interpolation_method=interpolation_method,
        )

    elif method.lower() in ["cross-correlation", "xcorr"]:
        rate = _rsp_rate_xcorr(
            rsp_cleaned,
            sampling_rate=sampling_rate,
            window=window,
            hop_size=hop_size,
            interpolation_method=interpolation_method,
        )

    else:
        raise ValueError("NeuroKit error: rsp_rate(): 'method' should be" " one of 'trough', or 'cross-correlation'.")

    return rate


# =============================================================================
# Cross-correlation method
# =============================================================================


def _rsp_rate_xcorr(rsp_cleaned, sampling_rate=1000, window=10, hop_size=1, interpolation_method="monotone_cubic"):

    N = len(rsp_cleaned)
    # Downsample data to 10Hz
    desired_sampling_rate = 10
    rsp = signal_resample(rsp_cleaned, sampling_rate=sampling_rate, desired_sampling_rate=desired_sampling_rate)

    # Define paramters
    window_length = int(desired_sampling_rate * window)

    rsp_rate = []
    for start in np.arange(0, N, hop_size):
        window_segment = rsp[start : start + window_length]
        if len(window_segment) < window_length:
            break  # the last frames that are smaller than windlow_length
        # Calculate the 1-order difference
        diff = np.ediff1d(window_segment)
        norm_diff = diff / np.max(diff)
        # Find xcorr for all frequencies with diff
        xcorr = []
        t = np.linspace(0, window, len(diff))
        for frequency in np.arange(5 / 60, 30.25 / 60, 0.25 / 60):
            # Define the sin waves
            sin_wave = np.sin(2 * np.pi * frequency * t)
            # Calculate cross-correlation
            _xcorr = np.corrcoef(norm_diff, sin_wave)[0, 1]
            xcorr.append(_xcorr)

        # Find frequency with the highest xcorr with diff
        max_frequency_idx = np.argmax(xcorr)
        max_frequency = np.arange(5 / 60, 30.25 / 60, 0.25 / 60)[max_frequency_idx]
        # Append max_frequency to rsp_rate - instanteneous rate
        rsp_rate.append(max_frequency)

    x = np.arange(len(rsp_rate))
    y = rsp_rate
    rsp_rate = signal_interpolate(x, y, x_new=len(rsp_cleaned), method=interpolation_method)
    # Smoothing
    rsp_rate = signal_filter(rsp_rate, highcut=0.1, order=4, sampling_rate=sampling_rate)

    # Convert to Brpm
    rsp_rate = np.multiply(rsp_rate, 60)

    return np.array(rsp_rate)
