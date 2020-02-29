# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .signal_interpolate import signal_interpolate
from .signal_formatpeaks import _signal_formatpeaks_sanitize



def signal_rate(peaks, sampling_rate=1000, desired_length=None):
    """Calculate signal rate from a series of peaks.

    Parameters
    ----------
    peaks : list, array, DataFrame, Series or dict
        The samples at which thepeaks occur. If an array is
        passed, it is assumed that these containers were obtained with
        `signal_findpeaks()`. If a DataFrame is passed, it is assumed it is of the same length as
        the input signal in which occurrences of R-peaks are marked as "1", with such containers
        obtained with e.g., ecg_findpeaks() or rsp_findpeaks().
    sampling_rate : int
        The sampling frequency of the signal that contains the R-peaks (in Hz,
        i.e., samples/second). Defaults to 1000.
    desired_length : int
        By default, the returned signal rate has the same number of elements as
        the raw signal. If set to an integer, the returned signal rate will be
        interpolated between R-peaks over `desired_length` samples. Has no
        effect if a DataFrame is passed in as the `signal` argument. Defaults to
        None.

    Returns
    -------
    array
        A vector containing the rate.

    See Also
    --------
    signal_findpeaks, signal_fixpeaks, signal_plot, rsp_rate, ecg_rate

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=10, sampling_rate=1000, frequency=1)
    >>> info = nk.signal_findpeaks(signal)
    >>>
    >>> rate = nk.signal_rate(peaks=info["Peaks"])
    >>> nk.signal_plot(rate)
    """

    period = _signal_period(peaks, sampling_rate, desired_length)
    rate = 60 / period

    return rate







def _signal_period(peaks, sampling_rate=1000, desired_length=None):
    """
    Return the peak interval in seconds.
    """
    # Format input.
    peaks, desired_length = _signal_formatpeaks_sanitize(peaks, desired_length)

    # Sanity checks.
    if len(peaks) <= 3:
        print("NeuroKit warning: _signal_formatpeaks(): too few peaks detected to "
              "compute the rate. Returning empty vector.")
        return np.full(desired_length, np.nan)

    # Calculate period in msec, based on peak to peak difference and make sure
    # that rate has the same number of elements as peaks (important for
    # interpolation later) by prepending the mean of all periods.
    period = np.ediff1d(peaks, to_begin=0) / sampling_rate
    period[0] = np.mean(period[1:])

    # Interpolate all statistics to desired length.
    if desired_length != np.size(peaks):
        period = signal_interpolate(peaks, period, desired_length=desired_length)

    return period
