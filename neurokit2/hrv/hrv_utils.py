# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .intervals_to_peaks import intervals_to_peaks
from .intervals_utils import _intervals_sanitize


def _hrv_get_rri(peaks=None, sampling_rate=1000):
    if peaks is None:
        return None, None, None
    # Compute R-R intervals (also referred to as NN) in milliseconds
    rri = np.diff(peaks) / sampling_rate * 1000
    rri, rri_time, rri_missing = _intervals_sanitize(rri)
    return rri, rri_time, rri_missing


def _hrv_format_input(peaks=None, sampling_rate=1000, output_format="intervals"):

    if isinstance(peaks, tuple):
        rri, rri_time, rri_missing, sampling_rate = _hrv_sanitize_tuple(peaks, sampling_rate=sampling_rate)
    elif isinstance(peaks, (dict, pd.DataFrame)):
        rri, rri_time, rri_missing, sampling_rate = _hrv_sanitize_dict_or_df(peaks, sampling_rate=sampling_rate)
    else:
        peaks = _hrv_sanitize_peaks(peaks)
        rri, rri_time, rri_missing = _hrv_get_rri(peaks, sampling_rate=sampling_rate)
    if output_format == "intervals":
        return rri, rri_time, rri_missing
    elif output_format == "peaks":
        return (
            intervals_to_peaks(rri, intervals_time=rri_time, sampling_rate=sampling_rate),
            sampling_rate,
        )


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

    rri, rri_time, rri_missing = _hrv_get_rri(peaks=peaks, sampling_rate=sampling_rate)

    return rri, rri_time, rri_missing, sampling_rate


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
        rri, rri_time, rri_missing = _intervals_sanitize(rri, intervals_time=rri_time)
        return rri, rri_time, rri_missing, sampling_rate

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
        rri, rri_time, rri_missing = _hrv_get_rri(peaks=peaks, sampling_rate=sampling_rate)
    else:
        rri, rri_time, rri_missing = _hrv_get_rri(peaks=peaks)
    return rri, rri_time, rri_missing, sampling_rate


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
