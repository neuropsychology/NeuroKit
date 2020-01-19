# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .signal_formatpeaks import _signal_formatpeaks
from .signal_rate import _signal_period
from ..stats import standardize


def signal_fixpeaks(peaks, sampling_rate=1000, interval_min=None, interval_max=None, relative_interval_min=None, relative_interval_max=None, robust=False):
    """Detect and correct outliers in peaks.

    Parameters
    ----------
    peaks : list, array, DataFrame, Series or dict
        The samples at which thepeaks occur. If an array is
        passed, it is assumed that these containers were obtained with
        `signal_findpeaks()`. If a DataFrame is passed, it is assumed it is of the same length as
        the input signal in which occurrences of R-peaks are marked as "1", with such containers
        obtained with e.g., ecg_findpeaks() or rsp_findpeaks().
    sampling_rate : int
        The sampling frequency of the signal that contains the R-peaks (in Hz,
        i.e., samples/second). Defaults to 1000.
    interval_min, interval_max : float
        The minimum or maximum interval between the peaks.
    relative_interval_min, relative_interval_max : float
        The minimum or maximum interval between the peaks as relative to the sample (expressed in standard deviation from the mean).
    robust : bool
        Use a robust method of standardization (see `standardize()`) for the relative thresholds.


    Returns
    -------
    array
        A vector containing the corrected peaks.

    See Also
    --------
    signal_findpeaks, signal_rate, standardize

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=4, sampling_rate=1000, frequency=1)
    >>> peaks_true = nk.signal_findpeaks(signal)["Peaks"]
    >>> peaks = np.delete(peaks_true, [1])  # Create wholes

    >>> signal = nk.signal_simulate(duration=20, sampling_rate=1000, frequency=1)
    >>> peaks_true = nk.signal_findpeaks(signal)["Peaks"]
    >>> peaks = np.delete(peaks_true, [5, 15])  # Create wholes
    >>> peaks = np.sort(np.append(peaks, [1350, 11350, 18350]))  # Add artifacts
    >>>
    >>> peaks_corrected = nk.signal_fixpeaks(peaks=peaks, interval_min=0.5, interval_max=1.5)
    >>> nk.events_plot([peaks_corrected, peaks], signal)
    """
    # Format input.
    peaks, desired_length = _signal_formatpeaks(peaks)


    # Minimum
    peaks = _signal_fixpeaks_remove_small(peaks, sampling_rate,
                                          interval_min, relative_interval_min, robust)


    # Maximum
    peaks = _signal_fixpeaks_interpolate_big(peaks, sampling_rate,
                                             interval_max, relative_interval_max, robust)


    return peaks



# =============================================================================
# Utilities
# =============================================================================

def _signal_fixpeaks_remove_small(peaks, sampling_rate=1000, interval_min=None, relative_interval_min=None, robust=False):
    if interval_min is None and relative_interval_min is None:
        return peaks

    if interval_min is not None:
        interval = _signal_period(peaks, sampling_rate=sampling_rate, desired_length=len(peaks))
        peaks = peaks[interval > interval_min]

    if relative_interval_min is not None:
        interval = _signal_period(peaks, sampling_rate=sampling_rate, desired_length=len(peaks))
        peaks = peaks[standardize(interval, robust=robust) > relative_interval_min]

    return peaks



def _signal_fixpeaks_interpolate_big(peaks, sampling_rate=1000, interval_max=None, relative_interval_max=None, robust=False):
    if interval_max is None and relative_interval_max is None:
        return peaks

    continue_loop = True
    while continue_loop is True:
        if interval_max is not None:
            interval = _signal_period(peaks, sampling_rate=sampling_rate, desired_length=len(peaks))
            peaks, continue_loop = _signal_fixpeaks_interpolate_missing(peaks, interval, interval_max, sampling_rate)

        if relative_interval_max is not None:
            interval = _signal_period(peaks, sampling_rate=sampling_rate, desired_length=len(peaks))
            peaks, continue_loop = _signal_fixpeaks_interpolate_missing(peaks, standardize(interval, robust=robust), interval_max, sampling_rate)

    return peaks





def _signal_fixpeaks_interpolate_missing(peaks, interval, interval_max, sampling_rate):
    outliers = interval > interval_max
    outliers_loc = np.where(outliers)[0]
    if np.sum(outliers) == 0:
        return peaks, False

    # Delete large interval and replace by two unknown intervals
    interval[outliers] = np.nan
    interval = np.insert(interval, outliers_loc, np.nan)
#    new_peaks_location = np.where(np.isnan(interval))[0]

    # Interpolate values
    interval = pd.Series(interval).interpolate().values
    peaks_corrected = _signal_fixpeaks_period_to_location(interval, sampling_rate, first_location=peaks[0])
    peaks = np.insert(peaks, outliers_loc, peaks_corrected[outliers_loc + np.arange(len(outliers_loc))])
    return peaks, True




def _signal_fixpeaks_period_to_location(period, sampling_rate=1000, first_location=0):
    """
    """
    location = np.cumsum(period * sampling_rate)
    location = location - (location[0] - first_location)
    return location.astype(np.int)