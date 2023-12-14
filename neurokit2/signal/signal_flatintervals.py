# -*- coding: utf-8 -*-
import numpy as np


def signal_flatintervals(signal, sampling_rate, threshold=0.01, tolerance=60):
    """Finds flatline areas in a signal.

    Parameters
    ----------
    signal : Union[list, np.array]
        The signal as a vector of values.
    sampling_rate : int
        The sampling rate of the signal, i.e. how many samples (values) per second.
    threshold : float, optional
        Flatline threshold relative to the biggest change in the signal.
        This is the percentage of the maximum value of absolute consecutive differences.
        Default: 0.01 (= 1% of the biggest change in the signal)
    tolerance : int, optional
        Determines how fine-grained the resulting signal is,
        i.e. how long (in seconds) can a flatline part be without being recognised as such.
        Default: 60 (seconds)

    Returns
    -------
    list
        Returns a list of tuples:
        [(flatline_starts1, flatline_ends1), (flatline_starts2, flatline_ends2), ...]
        flatline_starts: Index where a flatline part starts.
        flatline_ends: Index where a flatline part ends.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      ecg = nk.ecg_simulate(duration=10 * one_minute, sampling_rate=sampling_rate)
      flatline_1 = np.full(10 * one_minute * sampling_rate, -4.0)
      flatline_2 = np.zeros(10 * one_minute * sampling_rate)
      signal = np.concatenate([ecg, flatline_1, ecg, flatline_2, ecg, flatline_1])

      nk.signal_flatintervals(signal)
    
    """

    # Identify flanks: +1 for beginning plateau; -1 for ending plateau.
    flanks = np.diff(_find_flatlines(signal, sampling_rate, threshold, tolerance).astype(int))

    flatline_starts = np.flatnonzero(flanks > 0)
    flatline_ends = np.flatnonzero(flanks < 0)

    # Correct offsets from moving average
    flatline_starts = flatline_starts + sampling_rate * tolerance

    # Insert start marker at signal start if a start marker is missing
    if len(flatline_starts) < len(flatline_ends):
        flatline_starts = np.insert(flatline_starts, 0, 0)

    # Insert end marker at signal end if an end marker is missing
    if len(flatline_ends) < len(flatline_starts):
        flatline_ends = np.append(flatline_ends, [len(signal) - 1])

    # Return instances where start < end (start >= end might occur due to offset correction).
    return [(start, end) for start, end in zip(flatline_starts, flatline_ends) if start < end]


def _find_flatlines(signal, sampling_rate, threshold=0.01, tolerance=60):
    """Finds flatline areas in a signal.

    Parameters
    ----------
    signal : Union[list, np.array]
        The signal as a vector of values.
    sampling_rate : int
        The sampling rate of the signal, i.e. how many samples (values) per second.
    threshold : float, optional
        Flatline threshold relative to the biggest change in the signal.
        This is the percentage of the maximum value of absolute consecutive differences.
        Default: 0.01 (= 1% of the biggest change in the signal)
    tolerance : int, optional
        Determines how fine-grained the resulting signal is,
        i.e. how long (in seconds) can a flatline part be without being recognised as such.
        Default: 60 (seconds)

    Returns
    -------
    np.array
        Returns a signal that is True/1 where there is a sufficiently long
        flatline part in the signal and False/0 otherwise.
        Note: Returned signal is shorter than the original signal by (sampling_rate * tolerance) - 1.

    """
    abs_diff = np.abs(np.diff(signal))
    threshold = threshold * np.max(abs_diff)

    return _moving_average(abs_diff >= threshold, sampling_rate * tolerance) < threshold


def _moving_average(signal, window_size):
    """Moving window average on a signal.

    Parameters
    ----------
    signal : Union[list, np.array]
        The signal as a vector of values.
    window_size : int
        How many consequtive samples are used for averaging.

    Returns
    -------
    np.array
        Returns a signal of averages from the original signal.
        Note: The returned signal is shorter than the original signal by window_size - 1.

    """

    return np.convolve(signal, np.ones(window_size), "valid") / window_size
