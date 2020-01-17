# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import scipy.signal
import scipy.misc

from ..stats import standardize
from .signal_zerocrossings import signal_zerocrossings
from ..misc import findclosest


def signal_findpeaks(signal, height_min=None, width_min=None, height_max=None, width_max=None, relative_height_min=None, relative_width_min=None, relative_height_max=None, relative_width_max=None):
    """Find peaks in a signal.

    Locate peaks (local maxima) in a signal and their related characteristics, such as height (prominence), width and distance with other peaks.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    distance_min, height_min, width_min, distance_max, height_max, width_max : float
        The minimum or maximum distance (between peaks, in number of sample points), height (i.e., amplitude in terms of absolute values) or width of the peaks (in number of sample points). For example, `distance_min=20` will remove all peaks which distance with the previous peak is smaller or equal to 20 sample points.
    relative_distance_min, relative_height_min, relative_width_min, relative_distance_max, relative_height_max, relative_width_max : float
        The minimum or maximum distance (between peaks), height (i.e., amplitude) or width of the peaks in terms of standard deviation from the sample. For example, `relative_distance_min=-2.96` will remove all peaks which distance lies below 2.96 standard deviations from the mean of the distances.

    Returns
    ----------
    dict
        Returns a dict itself containing 5 arrays:
            - 'Peaks' contains the peaks indices (as relative to the given signal). For instance, the value 3 means that the third data point of the signal is a peak.
            - 'Distance' contains, for each peak, the closest distance with another peak. Note that these values will be recomputed after filtering to match the selected peaks.
            - 'Height' contains the prominence of each peak. See `scipy.signal.peak_prominences()`.
            - 'Width' contains the width of each peak. See `scipy.signal.peak_widths()`.
            - 'Onset' contains the onset, start (or left trough), of each peak. See `scipy.signal.peak_widths()`.
            - 'Offset' contains the offset, end (or right trough), of each peak. See `scipy.signal.peak_widths()`.

    Examples
    ---------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>> import scipy.misc
    >>>
    >>> signal = nk.signal_simulate(duration=5)
    >>> info = nk.signal_findpeaks(signal)
    >>> nk.events_plot([info["Onset"], info["Peaks"]], signal)
    >>>
    >>> signal = nk.signal_distord(signal)
    >>> info = nk.signal_findpeaks(signal, height_min=1, width_min=2)
    >>> nk.events_plot(info["Peaks"], signal)
    >>>
    >>> # Filter peaks
    >>> ecg = scipy.misc.electrocardiogram()
    >>> signal = ecg[0:1000]
    >>> info1 = nk.signal_findpeaks(signal, relative_height_min=0)
    >>> info2 = nk.signal_findpeaks(signal, relative_height_min=1)
    >>> nk.events_plot([info1["Peaks"], info2["Peaks"]], signal)

    See Also
    --------
    scipy.signal.find_peaks, scipy.signal.peak_widths, peak_prominences.signal.peak_widths
    """
    info = _signal_findpeaks_scipy(signal)

    keep = np.full(len(info["Peaks"]), True)

    # Absolute indices - min
    if height_min is not None:
        keep[info["Height"] < height_min] = False
    if width_min is not None:
        keep[info["Width"] < width_min] = False

    # Absolute indices - max
    if height_max is not None:
        keep[info["Height"] > height_max] = False
    if width_max is not None:
        keep[info["Width"] > width_max] = False

    # Relative indices - min
    if relative_height_min is not None:
        keep[standardize(info["Height"]) < relative_height_min] = False
    if relative_width_min is not None:
        keep[standardize(info["Width"]) < relative_width_min] = False

    # Relative indices - max
    if relative_height_max is not None:
        keep[standardize(info["Height"]) > relative_height_max] = False
    if relative_width_max is not None:
        keep[standardize(info["Width"]) > relative_width_max] = False

    # Filter
    info["Peaks"] = info["Peaks"][keep]
    info["Distance"] = _signal_findpeaks_distances(info["Peaks"])
    info["Height"] = info["Height"][keep]
    info["Width"] = info["Width"][keep]
    info["Onset"] = _signal_findpeaks_base(info["Peaks"], signal, what="onset")
    info["Offset"] = _signal_findpeaks_base(info["Peaks"], signal, what="offset")

    return info





# =============================================================================
# Internals
# =============================================================================


def _signal_findpeaks_distances(peaks):

    if len(peaks) <= 2:
        distances = np.full(len(peaks), np.nan)
    else:
        distances_next = np.concatenate([[np.nan], np.abs(np.diff(peaks))])
        distances_prev = np.concatenate([np.abs(np.diff(peaks[::-1])), [np.nan]])
        distances = np.array([np.nanmin(i) for i in list(zip(distances_next, distances_prev))])

    return distances



def _signal_findpeaks_base(peaks, signal, what="onset"):
    if what == "onset":
        direction = "smaller"
    else:
        direction = "greater"

    # Compute gradient (sort of derivative)
    gradient = np.gradient(signal)

    # Find zero-crossings
    zeros = signal_zerocrossings(gradient)

    onsets = np.zeros(len(peaks), np.int)
    for i, peak in enumerate(peaks):
        onsets[i] = findclosest(peak, zeros, direction=direction, strictly=True)

    return onsets



def _signal_findpeaks_scipy(signal):
    peaks, _ = scipy.signal.find_peaks(signal)

    # Get info
    distances = _signal_findpeaks_distances(peaks)
    heights, left_base, right_base = scipy.signal.peak_prominences(signal, peaks)
    widths, width_heights, left_ips, right_ips = scipy.signal.peak_widths(signal, peaks, rel_height=0.5)

    # Prepare output
    info = {"Peaks": peaks,
            "Distance": distances,
            "Height": heights,
            "Width": widths}

    return info
