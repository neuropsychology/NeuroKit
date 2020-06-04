# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal

from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize


def eda_fixpeaks(peaks, onsets=None, height=None, sampling_rate=1000):
    """
    Correct  Skin Conductance Responses (SCR) peaks.

    Low-level function used by `eda_peaks()` to correct the peaks found by `eda_findpeaks()`. Doesn't do anything for now for EDA. See `eda_peaks()` for details.

    Parameters
    ----------
    peaks, onsets, height : list, array, DataFrame, Series or dict
        The samples at which the SCR peaks or its onset occur (or the amplitude of the peak). If a dict or a
        DataFrame is passed, it is assumed that these containers were obtained
        with `eda_findpeaks()`.
    sampling_rate : int
        The sampling frequency of the signal that contains the peaks (in Hz,
        i.e., samples/second).

    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the
        aplitude of the SCR, the samples at which the SCR onset and the
        SCR peaks occur. Accessible with the keys "SCR_Amplitude", "SCR_Onsets",
        and "SCR_Peaks" respectively.

    See Also
    --------
    eda_simulate, eda_clean, eda_phasic, eda_findpeaks, eda_peaks, eda_process, eda_plot



    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> # Get phasic component
    >>> eda_signal = nk.eda_simulate(duration=30, scr_number=5, drift=0.1, noise=0)
    >>> eda_cleaned = nk.eda_clean(eda_signal)
    >>> eda = nk.eda_phasic(eda_cleaned)
    >>> eda_phasic = eda["EDA_Phasic"].values
    >>>
    >>> # Find and fix peaks
    >>> info = nk.eda_findpeaks(eda_phasic)
    >>> info = nk.eda_fixpeaks(info)
    >>>
    >>> fig = nk.events_plot(info["SCR_Peaks"], eda_phasic)
    >>> fig #doctest: +SKIP

    """
    # Format input.
    peaks, onsets, height, desired_length = _eda_fixpeaks_retrieve(peaks, onsets, height, desired_length=None)

    # Do whatever fixing is required (nothing for now)

    # Prepare output
    info = {"SCR_Onsets": onsets, "SCR_Peaks": peaks, "SCR_Height": height}
    return info


# =============================================================================
# Internals
# =============================================================================
def _eda_fixpeaks_retrieve(peaks, onsets=None, height=None, desired_length=None):
    # Format input.
    original_input = peaks
    peaks, desired_length = _signal_formatpeaks_sanitize(original_input, desired_length, key="Peaks")
    if onsets is None:
        onsets, _ = _signal_formatpeaks_sanitize(original_input, desired_length, key="Onsets")
    if height is None:
        height, _ = _signal_formatpeaks_sanitize(original_input, desired_length, key="Height")
    return peaks, onsets, height, desired_length
