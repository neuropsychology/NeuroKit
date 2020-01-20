# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal.signal_rate import signal_rate
from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize
from ..signal import signal_interpolate
from ..signal import signal_smooth





def rsp_fixpeaks(peaks, troughs=None, sampling_rate=1000):
    """Correct and format RSP peaks.

    Compute respiration rate with the specified method.

    Parameters
    ----------
    peaks, troughs : list, array, DataFrame, Series or dict
        The samples at which the inhalation peaks occur. If a dict or a
        DataFrame is passed, it is assumed that these containers were obtained
        with `rsp_findpeaks()`.
    desired_length : int
        By default, the returned respiration rate has the same number of
        elements as `peaks`. If set to an integer, the returned rate will be
        interpolated between `peaks` over `desired_length` samples. Has no
        effect if a DataFrame is passed in as the `peaks` argument.
    sampling_rate : int
        The sampling frequency of the signal that contains the peaks (in Hz,
        i.e., samples/second).

    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the
        samples at which inhalation peaks and exhalation troughs occur,
        accessible with the keys "RSP_Peaks", and "RSP_Troughs", respectively.

    See Also
    --------
    rsp_clean, rsp_findpeaks, rsp_peaks, rsp_amplitude, rsp_process, rsp_plot

    Examples
    --------
    >>>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(duration=30, respiratory_rate=15)
    >>> cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    >>> info = nk.rsp_findpeaks(cleaned)
    >>> info = nk.rsp_fixpeaks(info)
    >>> nk.events_plot([info["RSP_Peaks"], info["RSP_Troughs"]], cleaned)
    """
    # Format input.
    peaks, troughs, desired_length = _rsp_fixpeaks_retrieve(peaks, troughs, desired_length=None)

    # Do whatever fixing is required (nothing for now)

    # Prepare output
    info = {"RSP_Peaks": peaks,
            "RSP_Troughs": troughs}

    return info


# =============================================================================
# Internals
# =============================================================================
def _rsp_fixpeaks_retrieve(peaks, troughs=None, desired_length=None):
    # Format input.
    original_input = peaks
    peaks, desired_length = _signal_formatpeaks_sanitize(original_input, desired_length, key="Peaks")
    if troughs is None:
        troughs, _ = _signal_formatpeaks_sanitize(original_input, desired_length, key="Troughs")
    return peaks, troughs, desired_length
