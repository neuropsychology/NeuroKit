# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np

from ..misc import NeuroKitWarning
from .signal_formatpeaks import _signal_formatpeaks_sanitize
from .signal_interpolate import signal_interpolate


def signal_period(peaks, sampling_rate=1000, desired_length=None, interpolation_method="monotone_cubic"):
    """Calculate signal period from a series of peaks.

    Parameters
    ----------
    peaks : Union[list, np.array, pd.DataFrame, pd.Series, dict]
        The samples at which the peaks occur. If an array is passed in, it is assumed that it was obtained
        with `signal_findpeaks()`. If a DataFrame is passed in, it is assumed it is of the same length as
        the input signal in which occurrences of R-peaks are marked as "1", with such containers obtained
        with e.g., ecg_findpeaks() or rsp_findpeaks().
    sampling_rate : int
        The sampling frequency of the signal that contains peaks (in Hz, i.e., samples/second).
        Defaults to 1000.
    desired_length : int
        If left at the default None, the returned period will have the same number of elements as peaks.
        If set to a value larger than the sample at which the last peak occurs in the signal (i.e., peaks[-1]),
        the returned period will be interpolated between peaks over `desired_length` samples. To interpolate
        the period over the entire duration of the signal, set desired_length to the number of samples in the
        signal. Cannot be smaller than or equal to the sample at which the last peak occurs in the signal.
        Defaults to None.
    interpolation_method : str
        Method used to interpolate the rate between peaks. See `signal_interpolate()`. 'monotone_cubic' is chosen
        as the default interpolation method since it ensures monotone interpolation between data points
        (i.e., it prevents physiologically implausible "overshoots" or "undershoots" in the y-direction).
        In contrast, the widely used cubic spline interpolation does not ensure monotonicity.
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
    >>> period = nk.signal_period(peaks=info["Peaks"], desired_length=len(signal))
    >>> nk.signal_plot(period)

    """
    peaks = _signal_formatpeaks_sanitize(peaks)

    # Sanity checks.
    if np.size(peaks) <= 3:
        warn(
            "Too few peaks detected to compute the rate. Returning empty vector.",
            category=NeuroKitWarning
        )
        return np.full(desired_length, np.nan)

    if isinstance(desired_length, (int, float)):
        if desired_length <= peaks[-1]:
            raise ValueError("NeuroKit error: desired_length must be None or larger than the index of the last peak.")

    # Calculate period in sec, based on peak to peak difference and make sure
    # that rate has the same number of elements as peaks (important for
    # interpolation later) by prepending the mean of all periods.
    period = np.ediff1d(peaks, to_begin=0) / sampling_rate
    period[0] = np.mean(period[1:])

    # Interpolate all statistics to desired length.
    if desired_length is not None:
        period = signal_interpolate(peaks, period, x_new=np.arange(desired_length), method=interpolation_method)

    return period
