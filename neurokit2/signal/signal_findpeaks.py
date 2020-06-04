# -*- coding: utf-8 -*-
import numpy as np
import scipy.misc
import scipy.signal

from ..misc import as_vector, find_closest
from ..stats import standardize
from .signal_zerocrossings import signal_zerocrossings


def signal_findpeaks(
    signal,
    height_min=None,
    height_max=None,
    relative_height_min=None,
    relative_height_max=None,
    relative_mean=True,
    relative_median=False,
    relative_max=False,
):
    """
    Find peaks in a signal.

    Locate peaks (local maxima) in a signal and their related characteristics, such as height (prominence), width and distance with other peaks.

    Parameters
    ----------
    signal : list, array or Series
        The signal (i.e., a time series) in the form of a vector of values.
    height_min, distance_max : float
        The minimum or maximum height (i.e., amplitude in terms of absolute values). For example, `height_min=20` will remove all peaks which height is smaller or equal to 20 (in the provided signal's values).
    relative_height_min, relative_height_max : float
        The minimum or maximum height (i.e., amplitude) relative to the sample (see below).
    relative_mean, relative_median, relative_max : bool
        If a relative threshold is specified, how should it be computed (i.e., relative to what?). `relative_mean=True` will use Z-scores. For example, `relative_height_min=-2.96` will remove all peaks which height lies below 2.96 standard deviations from the mean of the heights. Relative to median uses a more robust form of standardization (see `standardize`), and relative to max will consider the maximum height as the reference.

    Returns
    ----------
    dict
        Returns a dict itself containing 5 arrays:
            - 'Peaks' contains the peaks indices (as relative to the given signal). For instance, the value 3 means that the third data point of the signal is a peak.
            - 'Distance' contains, for each peak, the closest distance with another peak. Note that these values will be recomputed after filtering to match the selected peaks.
            - 'Height' contains the prominence of each peak. See `scipy.signal.peak_prominences()`.
            - 'Width' contains the width of each peak. See `scipy.signal.peak_widths()`.
            - 'Onset' contains the onset, start (or left trough), of each peak.
            - 'Offset' contains the offset, end (or right trough), of each peak.

    Examples
    ---------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>> import scipy.misc
    >>>
    >>> signal = nk.signal_simulate(duration=5)
    >>> info = nk.signal_findpeaks(signal)
    >>> fig1 = nk.events_plot([info["Onsets"], info["Peaks"]], signal)
    >>> fig1 #doctest: +SKIP
    >>>
    >>> signal = nk.signal_distort(signal)
    >>> info = nk.signal_findpeaks(signal, height_min=1)
    >>> fig2 = nk.events_plot(info["Peaks"], signal)
    >>> fig2 #doctest: +SKIP
    >>>
    >>> # Filter peaks
    >>> ecg = scipy.misc.electrocardiogram()
    >>> signal = ecg[0:1000]
    >>> info1 = nk.signal_findpeaks(signal, relative_height_min=0)
    >>> info2 = nk.signal_findpeaks(signal, relative_height_min=1)
    >>> fig3 = nk.events_plot([info1["Peaks"], info2["Peaks"]], signal)
    >>> fig3 #doctest: +SKIP

    See Also
    --------
    scipy.signal.find_peaks, scipy.signal.peak_widths, peak_prominences.signal.peak_widths, eda_findpeaks, ecg_findpeaks, rsp_findpeaks, signal_fixpeaks

    """
    info = _signal_findpeaks_scipy(signal)

    # Absolute
    info = _signal_findpeaks_keep(
        info,
        what="Height",
        below=height_max,
        above=height_min,
        relative_mean=False,
        relative_median=False,
        relative_max=False,
    )

    # Relative
    info = _signal_findpeaks_keep(
        info,
        what="Height",
        below=relative_height_max,
        above=relative_height_min,
        relative_mean=relative_mean,
        relative_median=relative_median,
        relative_max=relative_max,
    )

    # Filter
    info["Distance"] = _signal_findpeaks_distances(info["Peaks"])
    info["Onsets"] = _signal_findpeaks_findbase(info["Peaks"], signal, what="onset")
    info["Offsets"] = _signal_findpeaks_findbase(info["Peaks"], signal, what="offset")

    return info


# =============================================================================
# Filtering peaks
# =============================================================================


def _signal_findpeaks_keep(
    info, what="Height", below=None, above=None, relative_mean=False, relative_median=False, relative_max=False
):

    if below is None and above is None:
        return info

    keep = np.full(len(info["Peaks"]), True)

    if relative_max is True:
        what = info[what] / np.max(info[what])
    elif relative_median is True:
        what = standardize(info[what], robust=True)
    elif relative_mean is True:
        what = standardize(info[what])
    else:
        what = info[what]

    if below is not None:
        keep[what > below] = False
    if above is not None:
        keep[what < above] = False

    info = _signal_findpeaks_filter(info, keep)
    return info


def _signal_findpeaks_filter(info, keep):
    for key in info.keys():
        info[key] = info[key][keep]

    return info


# =============================================================================
# Helpers
# =============================================================================


def _signal_findpeaks_distances(peaks):

    if len(peaks) <= 2:
        distances = np.full(len(peaks), np.nan)
    else:
        distances_next = np.concatenate([[np.nan], np.abs(np.diff(peaks))])
        distances_prev = np.concatenate([np.abs(np.diff(peaks[::-1])), [np.nan]])
        distances = np.array([np.nanmin(i) for i in list(zip(distances_next, distances_prev))])

    return distances


def _signal_findpeaks_findbase(peaks, signal, what="onset"):
    if what == "onset":
        direction = "smaller"
    else:
        direction = "greater"

    troughs, _ = scipy.signal.find_peaks(-1 * signal)

    bases = find_closest(peaks, troughs, direction=direction, strictly=True)
    bases = as_vector(bases)

    return bases


def _signal_findpeaks_scipy(signal):
    peaks, _ = scipy.signal.find_peaks(signal)

    # Get info
    distances = _signal_findpeaks_distances(peaks)
    heights, left_base, right_base = scipy.signal.peak_prominences(signal, peaks)
    widths, width_heights, left_ips, right_ips = scipy.signal.peak_widths(signal, peaks, rel_height=0.5)

    # Prepare output
    info = {"Peaks": peaks, "Distance": distances, "Height": heights, "Width": widths}

    return info
