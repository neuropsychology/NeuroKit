# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning, find_successive_intervals, intervals_to_peaks
from ..signal import signal_interpolate


def _hrv_preprocess_rri(rri, rri_time=None, interpolate=False, interpolation_rate=100, **kwargs):
    rri, rri_time = _hrv_sanitize_rri(rri, rri_time=rri_time)

    if interpolate is False:
        interpolation_rate = None

    else:
        # Rate should be at least 1 Hz (due to Nyquist & frequencies we are interested in)
        # We considered an interpolation rate 4 Hz by default to match Kubios
        # but in case of some applications with high heart rates we decided to make it 100 Hz
        # See https://github.com/neuropsychology/NeuroKit/pull/680 for more information
        # and if you have any thoughts to contribute, please let us know!
        if interpolation_rate < 1:
            warn(
                "The interpolation rate of the R-R intervals is too low for "
                " computing the frequency-domain features."
                " Consider increasing the interpolation rate to at least 1 Hz.",
                category=NeuroKitWarning,
            )

        # Compute x-values of interpolated heart period signal at requested sampling rate.
        x_new = np.arange(start=rri_time[0], stop=rri_time[-1] + 1 / interpolation_rate, step=1 / interpolation_rate,)

        rri = signal_interpolate(rri_time, rri, x_new=x_new, **kwargs)
    return rri, interpolation_rate


def _hrv_get_rri(peaks=None, sampling_rate=1000):
    if peaks is None:
        return None, None
    # Compute R-R intervals (also referred to as NN) in milliseconds
    rri = np.diff(peaks) / sampling_rate * 1000
    rri, rri_time = _hrv_sanitize_rri(rri)
    return rri, rri_time


def _hrv_sanitize_rri(rri, rri_time=None):
    if rri is None:
        return None, None
    else:
        # Ensure that input is numpy array
        rri = np.array(rri)
    if rri_time is None:
        # Compute the timestamps of the R-R intervals in seconds
        rri_time = np.nancumsum(rri / 1000)
    else:
        # Ensure that input is numpy array
        rri_time = np.array(rri_time)
        
        # Confirm that timestamps are in seconds
        successive_intervals = find_successive_intervals(rri, intervals_time=rri_time)

        if np.all(successive_intervals) is False:
            # If none of the differences between timestamps match
            # the length of the R-R intervals in seconds,
            # try converting milliseconds to seconds
            converted_successive_intervals = find_successive_intervals(rri, intervals_time=rri_time / 1000)

            # Check if converting to seconds increased the number of differences
            # between timestamps that match the length of the R-R intervals in seconds
            if len(converted_successive_intervals[converted_successive_intervals]) > len(
                successive_intervals[successive_intervals]
            ):
                # Assume timestamps were passed in milliseconds and convert to seconds
                rri_time = rri_time / 1000

    # Remove NaN R-R intervals, if any
    rri_time = rri_time[np.isfinite(rri)]
    rri = rri[np.isfinite(rri)]
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
        return intervals_to_peaks(rri, intervals_time=rri_time, sampling_rate=sampling_rate), sampling_rate


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
        rri, rri_time = _hrv_sanitize_rri(rri, rri_time=rri_time)
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
