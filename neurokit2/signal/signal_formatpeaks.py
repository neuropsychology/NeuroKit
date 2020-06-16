# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def signal_formatpeaks(info, desired_length, peak_indices=None):
    """Transforms an peak-info dict to a signal of given length."""
    if peak_indices is None:
        peak_indices = [key for key in info.keys() if "Peaks" in key]

    signals = {}
    for feature in info.keys():
        if any(x in str(feature) for x in ["Peak", "Onset", "Offset", "Trough", "Recovery"]):
            signals[feature] = _signal_from_indices(info[feature], desired_length, 1)
        else:
            signals[feature] = _signal_from_indices(peak_indices, desired_length, info[feature])
    signals = pd.DataFrame(signals)
    return signals


# =============================================================================
# Internals
# =============================================================================


def _signal_from_indices(indices, desired_length=None, value=1):
    """Generates array of 0 and given values at given indices.

    Used in *_findpeaks to transform vectors of peak indices to signal.

    """
    signal = np.zeros(desired_length)

    # Force indices as int
    if isinstance(indices[0], np.float):
        indices = indices[~np.isnan(indices)].astype(np.int)

    if isinstance(value, (int, float)):
        signal[indices] = value
    else:
        if len(value) != len(indices):
            raise ValueError(
                "NeuroKit error: _signal_from_indices(): The number of values "
                "is different from the number of indices."
            )
        signal[indices] = value
    return signal


def _signal_formatpeaks_sanitize(peaks, key="Peaks"):
    # Attempt to retrieve column.
    if isinstance(peaks, tuple):
        if isinstance(peaks[0], (dict, pd.DataFrame)):
            peaks = peaks[0]
        elif isinstance(peaks[1], dict):
            peaks = peaks[1]
        else:
            peaks = peaks[0]

    if isinstance(peaks, pd.DataFrame):
        col = [col for col in peaks.columns if key in col]
        if len(col) == 0:
            raise TypeError(
                "NeuroKit error: _signal_formatpeaks(): wrong type of input ",
                "provided. Please provide indices of peaks.",
            )
        peaks_signal = peaks[col[0]].values
        peaks = np.where(peaks_signal == 1)[0]

    if isinstance(peaks, dict):
        col = [col for col in list(peaks.keys()) if key in col]
        if len(col) == 0:
            raise TypeError(
                "NeuroKit error: _signal_formatpeaks(): wrong type of input ",
                "provided. Please provide indices of peaks.",
            )
        peaks = peaks[col[0]]

    # Retrieve length.
    try:  # Detect if single peak
        len(peaks)
    except TypeError:
        peaks = np.array([peaks])

    return peaks
