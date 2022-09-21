import numpy as np


def find_successive_intervals(intervals, intervals_time=None, thresh_unequal=2, n_diff=1):
    """Identify successive intervals.

    Identification of intervals that are consecutive
    (e.g. in case of missing data).

    Parameters
    ----------
    intervals : list or ndarray
        Intervals, e.g. breath-to-breath (BBI) or rpeak-to-rpeak (RRI)
    intervals_time : list or ndarray, optional
        Time points corresponding to intervals, in seconds.
    thresh_unequal : int or float, optional
        Threshold at which the difference between time points is considered to
        be unequal to the interval, in milliseconds.
    n_diff: int, optional
        The number of times values are differenced.
        Can be used to check which values are valid for the n-th difference
        assuming successive intervals.

    Returns
    ----------
    array
        An array of True/False with True being the successive intervals.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>> rri = [400, 500, 700, 800, 900]
    >>> rri_time = [0.7,  1.2,  2.5, 3.3, 4.2]
    >>> successive_intervals = nk.find_successive_intervals(rri, rri_time)
    >>> successive_intervals
    array([ True, False,  True,  True])
    >>> rri = [400, 500, np.nan, 700, 800, 900]
    >>> successive_intervals = find_successive_intervals(rri)
    >>> successive_intervals
    array([ True, False, False,  True,  True])

    """

    # Convert to numpy array
    intervals = np.array(intervals)

    if intervals_time is None:
        intervals_time = np.nancumsum(intervals / 1000)
    else:
        intervals_time = np.array(intervals_time)

    intervals_time[np.isnan(intervals)] = np.nan

    diff_intervals_time_ms = np.diff(intervals_time, n=n_diff) * 1000

    abs_error_intervals_ref_time = abs(diff_intervals_time_ms - np.diff(intervals[1:], n=n_diff - 1))

    successive_intervals = abs_error_intervals_ref_time <= thresh_unequal

    return np.array(successive_intervals)
