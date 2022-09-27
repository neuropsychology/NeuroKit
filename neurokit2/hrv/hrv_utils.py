# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .intervals_to_peaks import intervals_to_peaks


def _hrv_get_rri(peaks=None, sampling_rate=1000):
    if peaks is None:
        return None, None
    # Compute R-R intervals (also referred to as NN) in milliseconds
    rri = np.diff(peaks) / sampling_rate * 1000
    rri, rri_time = _intervals_sanitize(rri)
    return rri, rri_time


def _hrv_format_input(peaks=None, sampling_rate=1000, output_format="intervals"):

    if isinstance(peaks, tuple):
        rri, rri_time, sampling_rate = _hrv_sanitize_tuple(peaks, sampling_rate=sampling_rate)
    elif isinstance(peaks, (dict, pd.DataFrame)):
        rri, rri_time, sampling_rate = _hrv_sanitize_dict_or_df(peaks, sampling_rate=sampling_rate)
    else:
        peaks = _hrv_sanitize_peaks(peaks)
        rri, rri_time = _hrv_get_rri(peaks, sampling_rate=sampling_rate)
    if output_format == "intervals":
        return rri, rri_time
    elif output_format == "peaks":
        return (
            intervals_to_peaks(rri, intervals_time=rri_time, sampling_rate=sampling_rate),
            sampling_rate,
        )


def _intervals_successive(intervals, intervals_time=None, thresh_unequal=2, n_diff=1):
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
        intervals_time = np.array(intervals_time)

    intervals_time[np.isnan(intervals)] = np.nan

    diff_intervals_time_ms = np.diff(intervals_time, n=n_diff) * 1000

    abs_error_intervals_ref_time = abs(
        diff_intervals_time_ms - np.diff(intervals[1:], n=n_diff - 1)
    )

    successive_intervals = abs_error_intervals_ref_time <= thresh_unequal

    return np.array(successive_intervals)


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

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk
      ibi = [500, 400, 700, 500, 300, 800, 500]
      ibi, ibi_time = intervals_sanitize(ibi)

    """
    if intervals is None:
        return None, None
    else:
        # Ensure that input is numpy array
        intervals = np.array(intervals)
    if intervals_time is None:
        # Compute the timestamps of the intervals in seconds
        intervals_time = np.nancumsum(intervals / 1000)
    else:
        # Ensure that input is numpy array
        intervals_time = np.array(intervals_time)

        # Confirm that timestamps are in seconds
        successive_intervals = _intervals_successive(intervals, intervals_time=intervals_time)

        if np.all(successive_intervals) is False:
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

    if remove_missing:
        # Remove NaN R-R intervals, if any
        intervals_time = intervals_time[np.isfinite(intervals)]
        intervals = intervals[np.isfinite(intervals)]
    return intervals, intervals_time


# =============================================================================
# Internals
# =============================================================================
def _hrv_sanitize_tuple(peaks, sampling_rate=1000):

    # Get sampling rate
    info = [i for i in peaks if isinstance(i, dict)]
    sampling_rate = info[0]["sampling_rate"]

    # Detect actual sampling rate
    if len(info) < 1:
        peaks, sampling_rate = peaks[0], peaks[1]

    # Get peaks
    if isinstance(peaks[0], (dict, pd.DataFrame)):
        try:
            peaks = _hrv_sanitize_dict_or_df(peaks[0])
        except NameError:
            if isinstance(peaks[1], (dict, pd.DataFrame)):
                try:
                    peaks = _hrv_sanitize_dict_or_df(peaks[1])
                except NameError:
                    peaks = _hrv_sanitize_peaks(peaks[1])
            else:
                peaks = _hrv_sanitize_peaks(peaks[0])

    return _hrv_get_rri(peaks=peaks, sampling_rate=sampling_rate), sampling_rate


def _hrv_sanitize_dict_or_df(peaks, sampling_rate=None):

    # Get columns
    if isinstance(peaks, dict):
        cols = np.array(list(peaks.keys()))
        if "sampling_rate" in cols:
            sampling_rate = peaks["sampling_rate"]
    elif isinstance(peaks, pd.DataFrame):
        cols = peaks.columns.values

    # check whether R-R intervals were passed rather than peak indices
    if "RRI" in cols:
        rri = peaks["RRI"]
        if "RRI_Time" in cols:
            rri_time = peaks["RRI_Time"]
        else:
            rri_time = None
        rri, rri_time = _intervals_sanitize(rri, rri_time=rri_time)
        return rri, rri_time, sampling_rate

    cols = cols[["Peak" in s for s in cols]]

    if len(cols) > 1:
        cols = cols[[("ECG" in s) or ("PPG" in s) for s in cols]]

    if len(cols) == 0:
        raise NameError(
            "NeuroKit error: hrv(): Wrong input, ",
            "we couldn't extract R-peak indices. ",
            "You need to provide a list of R-peak indices.",
        )

    peaks = _hrv_sanitize_peaks(peaks[cols[0]])

    if sampling_rate is not None:
        rri, rri_time = _hrv_get_rri(peaks=peaks, sampling_rate=sampling_rate)
    else:
        rri, rri_time = _hrv_get_rri(peaks=peaks)
    return rri, rri_time, sampling_rate


def _hrv_sanitize_peaks(peaks):

    if isinstance(peaks, pd.Series):
        peaks = peaks.values

    if len(np.unique(peaks)) == 2:
        if np.all(np.unique(peaks) == np.array([0, 1])):
            peaks = np.where(peaks == 1)[0]

    if isinstance(peaks, list):
        peaks = np.array(peaks)

    if peaks is not None:
        if isinstance(peaks, tuple):
            if any(np.diff(peaks[0]) < 0):  # not continuously increasing
                raise ValueError(
                    "NeuroKit error: _hrv_sanitize_input(): "
                    + "The peak indices passed were detected as non-consecutive. You might have passed RR "
                    + "intervals instead of peaks. If so, convert RRIs into peaks using "
                    + "nk.intervals_to_peaks()."
                )
        else:
            if any(np.diff(peaks) < 0):
                raise ValueError(
                    "NeuroKit error: _hrv_sanitize_input(): "
                    + "The peak indices passed were detected as non-consecutive. You might have passed RR "
                    + "intervals instead of peaks. If so, convert RRIs into peaks using "
                    + "nk.intervals_to_peaks()."
                )

    return peaks
