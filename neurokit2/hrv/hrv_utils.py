# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from ..signal import signal_interpolate


def _hrv_get_rri(
    peaks=None, sampling_rate=1000, interpolate=False, interpolation_rate=100, **kwargs
):

    rri = np.diff(peaks) / sampling_rate * 1000

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
        x_new = np.arange(
            start=peaks[1] / sampling_rate,
            stop=peaks[-1] / sampling_rate + 1 / interpolation_rate,
            step=1 / interpolation_rate,
        )
        rri_time = (
            peaks[1:] / sampling_rate
        )  # Skip first peak since it has no corresponding element in heart_period

        rri = signal_interpolate(rri_time, rri, x_new=x_new, **kwargs)
    return rri, interpolation_rate


def _hrv_sanitize_input(peaks=None):

    if isinstance(peaks, tuple):
        peaks = _hrv_sanitize_tuple(peaks)
    elif isinstance(peaks, (dict, pd.DataFrame)):
        peaks = _hrv_sanitize_dict_or_df(peaks)
    else:
        peaks = _hrv_sanitize_peaks(peaks)

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


# =============================================================================
# Internals
# =============================================================================
def _hrv_sanitize_tuple(peaks):

    # Get sampling rate
    info = [i for i in peaks if isinstance(i, dict)]
    sampling_rate = info[0]["sampling_rate"]

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

    return peaks, sampling_rate


def _hrv_sanitize_dict_or_df(peaks):

    # Get columns
    if isinstance(peaks, dict):
        cols = np.array(list(peaks.keys()))
        if "sampling_rate" in cols:
            sampling_rate = peaks["sampling_rate"]
        else:
            sampling_rate = None
    elif isinstance(peaks, pd.DataFrame):
        cols = peaks.columns.values
        sampling_rate = None

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
        return peaks, sampling_rate
    else:
        return peaks


def _hrv_sanitize_peaks(peaks):

    if isinstance(peaks, pd.Series):
        peaks = peaks.values

    if len(np.unique(peaks)) == 2:
        if np.all(np.unique(peaks) == np.array([0, 1])):
            peaks = np.where(peaks == 1)[0]

    if isinstance(peaks, list):
        peaks = np.array(peaks)

    return peaks
