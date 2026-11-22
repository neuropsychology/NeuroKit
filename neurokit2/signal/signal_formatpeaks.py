import numpy as np
import pandas as pd


def signal_formatpeaks(info, desired_length, peak_indices=None, other_indices=None):
    """**Format Peaks**

    Transforms a peak-info dict to a signal of given length

    """
    if peak_indices is None:
        peaks_keys = [key for key in info.keys() if "Peaks" in key]
        if len(peaks_keys) > 0:
            peak_indices = info[peaks_keys[0]]

    signals = {}
    for feature, values in info.items():
        # Get indices of features
        if feature != "SCR_RecoveryTime" and any(x in str(feature) for x in ["Peak", "Onset", "Offset", "Trough", "Recovery"]):
            signals[feature] = _signal_from_indices(values, desired_length, 1)
            signals[feature] = signals[feature].astype("int64")  # indexing of feature using 1 and 0

        # Get values of features
        elif "RecoveryTime" in feature:
            # Sanitize indices and values without mutating `other_indices` across iterations
            sanitized_indices, other_values = _signal_sanitize_indices(other_indices, values)
            # Append recovery time values to signal
            signals[feature] = _signal_from_indices(sanitized_indices, desired_length, other_values)
        else:
            # Sanitize indices and values without mutating `peak_indices` across iterations
            sanitized_indices, peak_values = _signal_sanitize_indices(peak_indices, values)
            # Append peak values to signal
            signals[feature] = _signal_from_indices(sanitized_indices, desired_length, peak_values)

    signals = pd.DataFrame(signals)
    return signals


# =============================================================================
# Internals
# =============================================================================


def _signal_sanitize_indices(indices, values):
    if indices is None:
        return np.array([], dtype=int), np.array([])
    # Drop every nan index together with its value
    indices_arr = np.asarray(indices)
    nan_mask = pd.isna(indices_arr)
    if np.any(nan_mask):
        indices = indices_arr[~nan_mask]
        values = np.asarray(values)[~nan_mask]

    return indices, values


def _signal_from_indices(indices, desired_length=(), value=1):
    """**Generates array of 0 and given values at given indices**

    Used in *_findpeaks to transform vectors of peak indices to signal.

    """
    signal = pd.Series(np.zeros(desired_length, dtype=float))

    if isinstance(indices, list) and (not indices):  # skip empty lists
        return signal
    if isinstance(indices, np.ndarray) and (indices.size == 0):  # skip empty arrays
        return signal

    # Force indices as int, filtering any NaNs
    indices_arr = np.asarray(indices)
    if indices_arr.dtype.kind in ("f", "O"):
        valid = ~pd.isna(indices_arr)
        indices = indices_arr[valid].astype(int)
        if isinstance(value, (np.ndarray, list)) and len(value) == len(indices_arr):
            value = np.asarray(value)[valid]
    else:
        indices = indices_arr.astype(int)

    if len(indices) == 0:
        return signal

    # Appending single value
    if isinstance(value, (int, float)):
        signal.iloc[indices] = value
    # Appending multiple values
    elif isinstance(value, (np.ndarray, list)):
        if len(value) != len(indices):
            raise ValueError(
                "NeuroKit error: _signal_from_indices(): The number of values is different from the number of indices."
            )
        signal.iloc[indices] = value
    else:
        if len(value) != len(indices):
            raise ValueError(
                "NeuroKit error: _signal_from_indices(): The number of values is different from the number of indices."
            )
        signal.iloc[indices] = value

    return signal


def _signal_formatpeaks_sanitize(peaks, key="Peaks"):  # FIXME: private function not used in this module
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
