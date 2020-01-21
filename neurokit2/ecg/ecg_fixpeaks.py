# - * - coding: utf-8 - * -

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

from ..signal import signal_smooth
from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize


def ecg_fixpeaks(peaks, sampling_rate=1000):
    """Correct R-peaks location based on their interval (RRi).

    Low-level function used by `ecg_peaks()` to correct the peaks found by `ecg_findpeaks()`. See `ecg_peaks()` for details.

    Parameters
    ----------
    peaks, troughs : list, array, DataFrame, Series or dict
        The samples at which the inhalation peaks occur. If a dict or a
        DataFrame is passed, it is assumed that these containers were obtained
        with `ecg_findpeaks()`.
    sampling_rate : int
        The sampling frequency of the signal that contains the peaks (in Hz,
        i.e., samples/second).

    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the
        samples at which R-peaks occur, accessible with the key "ECG_R_Peaks".

    See Also
    --------
    ecg_clean, ecg_findpeaks, ecg_peaks, ecg_rate, ecg_process, ecg_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> ecg = nk.ecg_simulate(duration=10, sampling_rate=1000)
    >>> cleaned = nk.ecg_clean(ecg, sampling_rate=1000)
    >>>
    >>> # Extract and correct peaks
    >>> peak_info = nk.ecg_findpeaks(cleaned)
    >>> peak_info = nk.ecg_fixpeaks(peak_info)
    >>>
    >>> # Visualize
    >>> nk.events_plot(peak_info["ECG_R_Peaks"], cleaned)
    """
    # Format input.
    peaks, desired_length = _signal_formatpeaks_sanitize(peaks, desired_length=None)

    # Do whatever fixing is required (nothing for now)

    # Prepare output
    info = {"ECG_R_Peaks": peaks}

    return info
