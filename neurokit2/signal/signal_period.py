# -*- coding: utf-8 -*-
import numpy as np

from .signal_formatpeaks import _signal_formatpeaks_sanitize
from .signal_interpolate import signal_interpolate


def signal_period(peaks, sampling_rate=1000, desired_length=None, interpolation_order="cubic"):
    """
    Calculate signal period from a series of peaks.

    Parameters
    ----------
    peaks : list, array, DataFrame, Series or dict
        The samples at which the peaks occur. If an array is passed in, it is
        assumed that it was obtained with `signal_findpeaks()`. If a DataFrame
        is passed in, it is assumed it is of the same length as the input
        signal in which occurrences of R-peaks are marked as "1", with such
        containers obtained with e.g., ecg_findpeaks() or rsp_findpeaks().
    sampling_rate : int
        The sampling frequency of the signal that contains peaks (in Hz, i.e.,
        samples/second). Defaults to 1000.
    desired_length : int
        By default, the returned signal rate has the same number of elements as
        the raw signal. If set to an integer, the returned signal rate will be
        interpolated between peaks over `desired_length` samples. Has no
        effect if a DataFrame is passed in as the `signal` argument. Defaults
        to None.
    interpolation_order : str
        Order used to interpolate the rate between peaks. See
        `signal_interpolate()`.


    Returns
    -------
    array
        A vector containing the period.

    See Also
    --------
    signal_findpeaks, signal_fixpeaks, signal_plot

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
    peaks, desired_length = _signal_formatpeaks_sanitize(peaks, desired_length)

    # Sanity checks.
    if len(peaks) <= 3:
        print(
            "NeuroKit warning: _signal_formatpeaks(): too few peaks detected"
            " to compute the rate. Returning empty vector."
        )
        return np.full(desired_length, np.nan)

    # Calculate period in sec, based on peak to peak difference and make sure
    # that rate has the same number of elements as peaks (important for
    # interpolation later) by prepending the mean of all periods.
    period = np.ediff1d(peaks, to_begin=0) / sampling_rate
    period[0] = np.mean(period[1:])

    # Interpolate all statistics to desired length.
    if desired_length != np.size(peaks):
        period = signal_interpolate(peaks, period, desired_length=desired_length, method=interpolation_order)

    return period
