# -*- coding: utf-8 -*-
from .signal_period import signal_period


def signal_rate(peaks, sampling_rate=1000, desired_length=None, interpolation_order="cubic"):
    """
    Calculate signal rate from a series of peaks.

    This function can also be called either via ``ecg_rate()``, ```ppg_rate()`` or
    ``rsp_rate()`` (aliases provided for consistency).

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
        A vector containing the rate.

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
    >>> fig = nk.signal_plot(rate)
    >>> fig #doctest: +SKIP

    """
    period = signal_period(peaks, sampling_rate, desired_length, interpolation_order)
    rate = 60 / period

    return rate
