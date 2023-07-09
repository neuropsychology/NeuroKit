# -*- coding: utf-8 -*-
import numpy as np
import scipy


def _intervals_successive(intervals, intervals_time=None, thresh_unequal=10, n_diff=1):
    """Identify successive intervals.

    Identification of intervals that are consecutive
    (e.g. in case of missing data).

    Parameters
    ----------
    intervals : list or ndarray
        Intervals, e.g. breath-to-breath (BBI) or rpeak-to-rpeak (RRI)
    intervals_time : list or ndarray, optional
        Time points corresponding to intervals, in seconds. Defaults to None,
        in which case the cumulative sum of the intervals is used.
    thresh_unequal : int or float, optional
        Threshold at which the difference between time points is considered to
        be unequal to the interval, in milliseconds. Defaults to 10.
    n_diff: int, optional
        The number of times values are differenced.
        Can be used to check which values are valid for the n-th difference
        assuming successive intervals. Defaults to 1.

    Returns
    ----------
    array
        An array of True/False with True being the successive intervals.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      rri = [400, 500, 700, 800, 900]
      rri_time = [0.7,  1.2, 2.5, 3.3, 4.2]

      successive_intervals = nk.intervals_successive(rri, rri_time)
      successive_intervals

      rri = [400, 500, np.nan, 700, 800, 900]
      successive_intervals = nk.intervals_successive(rri)
      successive_intervals

    """

    # Convert to numpy array
    intervals = np.array(intervals)

    if intervals_time is None:
        intervals_time = np.nancumsum(intervals / 1000)
    else:
        intervals_time = np.array(intervals_time).astype(float)

    intervals_time[np.isnan(intervals)] = np.nan

    diff_intervals_time_ms = np.diff(intervals_time, n=n_diff) * 1000

    abs_error_intervals_ref_time = abs(
        diff_intervals_time_ms - np.diff(intervals[1:], n=n_diff - 1)
    )

    successive_intervals = abs_error_intervals_ref_time <= thresh_unequal

    return np.array(successive_intervals)


def _intervals_time_uniform(intervals_time, decimals=3):
    """Check whether timestamps are uniformly spaced.

    Useful for determining whether intervals have been interpolated.

    Parameters
    ----------
    intervals_time : list or array, optional
        List or numpy array of timestamps corresponding to intervals, in seconds.
    decimals : int, optional
        The precision of the timestamps. The default is 3.

    Returns
    ----------
    bool
        Whether the timestamps are uniformly spaced

    """
    return len(np.unique(np.round(np.diff(intervals_time), decimals=decimals))) == 1


def _intervals_sanitize(intervals, intervals_time=None, remove_missing=True):
    """**Interval input sanitization**

    Parameters
    ----------
    intervals : list or array
        List or numpy array of intervals, in milliseconds.
    intervals_time : list or array, optional
        List or numpy array of timestamps corresponding to intervals, in seconds.
    remove_missing : bool, optional
        Whether to remove NaNs and infinite values from intervals and timestamps.
        The default is True.

    Returns
    -------
    intervals : array
        Sanitized intervals, in milliseconds.
    intervals_time : array
        Sanitized timestamps corresponding to intervals, in seconds.
    intervals_missing : bool
        Whether there were missing intervals detected.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk
      ibi = [500, 400, 700, 500, 300, 800, 500]
      ibi, ibi_time, intervals_missing = intervals_sanitize(ibi)

    """
    if intervals is None:
        return None, None
    else:
        # Ensure that input is numpy array
        intervals = np.array(intervals)
    if intervals_time is None:
        # Impute intervals with median in case of missing values to calculate timestamps
        imputed_intervals = np.where(
            np.isnan(intervals), np.nanmedian(intervals, axis=0), intervals
        )
        # Compute the timestamps of the intervals in seconds
        intervals_time = np.nancumsum(imputed_intervals / 1000)
    else:
        # Ensure that input is numpy array
        intervals_time = np.array(intervals_time)

        # Confirm that timestamps are in seconds
        successive_intervals = _intervals_successive(intervals, intervals_time=intervals_time)

        if np.all(successive_intervals) is False:
            # Check whether intervals appear to be interpolated
            if not _intervals_time_uniform(intervals_time):
                # If none of the differences between timestamps match
                # the length of the R-R intervals in seconds,
                # try converting milliseconds to seconds
                converted_successive_intervals = _intervals_successive(
                    intervals, intervals_time=intervals_time / 1000
                )

                # Check if converting to seconds increased the number of differences
                # between timestamps that match the length of the R-R intervals in seconds
                if len(converted_successive_intervals[converted_successive_intervals]) > len(
                    successive_intervals[successive_intervals]
                ):
                    # Assume timestamps were passed in milliseconds and convert to seconds
                    intervals_time = intervals_time / 1000

    intervals_missing = _intervals_missing(intervals, intervals_time)

    if remove_missing:
        # Remove NaN R-R intervals, if any
        intervals_time = intervals_time[np.isfinite(intervals)]
        intervals = intervals[np.isfinite(intervals)]
    return intervals, intervals_time, intervals_missing


def _intervals_missing(intervals, intervals_time=None):
    if len(intervals[np.isfinite(intervals)]) < len(intervals):
        return True
    elif intervals_time is not None:
        successive_intervals = _intervals_successive(intervals, intervals_time=intervals_time)
        if not np.all(successive_intervals) and np.any(successive_intervals):
            # Check whether intervals appear to be interpolated
            if not _intervals_time_uniform(intervals_time):
                return True
    return False


def _intervals_time_to_sampling_rate(intervals_time, central_measure="mean"):
    """Get sampling rate from timestamps.

    Useful for determining sampling rate used to interpolate intervals.

    Parameters
    ----------
    intervals_time : list or array, optional
        List or numpy array of timestamps corresponding to intervals, in seconds.
    central_measure : str, optional
        The measure of central tendancy used. Either ``"mean"`` (default), ``"median"``, or ``"mode"``.

    Returns
    ----------
    bool
        Whether the timestamps are uniformly spaced

    """
    if central_measure == "mean":
        sampling_rate = float(1 / np.nanmean(np.diff(intervals_time)))
    elif central_measure == "median":
        sampling_rate = float(1 / np.nanmedian(np.diff(intervals_time)))
    else:
        sampling_rate = float(1 / scipy.stats.mode(np.diff(intervals_time)))
    return sampling_rate
