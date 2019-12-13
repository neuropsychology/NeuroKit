# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import scipy.signal
import scipy.misc

from ..stats import standardize





def _signal_findpeaks_distances(peaks):

    if len(peaks) <= 2:
        distances = np.full(len(peaks), np.nan)
    else:
        distances_next = np.concatenate([[np.nan], np.abs(np.diff(peaks))])
        distances_prev = np.concatenate([np.abs(np.diff(peaks[::-1])), [np.nan]])
        distances = np.array([np.nanmin(i) for i in list(zip(distances_next, distances_prev))])

    return(distances)






def _signal_findpeaks(signal):
    peaks, _ = scipy.signal.find_peaks(signal)

    # Get info
    distances = _signal_findpeaks_distances(peaks)
    heights, left_base, right_base = scipy.signal.peak_prominences(signal, peaks)
    widths, width_heights, left_ips, right_ips = scipy.signal.peak_widths(signal, peaks, rel_height=0.5)

    # Prepare output
    info = {"Distance": distances,
            "Height": heights,
            "Width": widths}

    return(peaks, info)








def signal_findpeaks(signal, distance_min=None, height_min=None, width_min=None, distance_max=None, height_max=None, width_max=None, relative_distance_min=None, relative_height_min=None, relative_width_min=None, relative_distance_max=None, relative_height_max=None, relative_width_max=None):
    """Find peaks in a signal.

    Locate peaks (local maxima) in a signal and their related characteristics, such as height (prominence), width and distance with other peaks.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.

    Returns
    ----------
    array, dict
        Returns two things. An array containing the peaks indices (as relative to the given signal).
        For instance, the value 3 means that the third sample of the signal is a peak or a troughs.
        It also returns a dict itself containing 3 arrays:
            - 'Distance' contains, for each peak, the closest distance with another peak. Note that these values will be recomputed after filtering to match the selected peaks.
            - 'Height' contains the prominence of each peak. See `scipy.signal.peak_prominences()`.
            - 'Width' contains the width of each peak. See `scipy.signal.peak_widths()`.

    Examples
    ---------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>> import scipy.misc
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=30, num=1000))
    >>> pd.Series(signal).plot()
    >>> peaks, info = nk.signal_findpeaks(signal)
    >>> nk.plot_events_in_signal(signal, peaks)
    >>>
    >>> signal = np.concatenate([np.arange(0, 20, 0.1), np.arange(17, 30, 0.1), np.arange(30, 10, -0.1)])
    >>> peaks, info = nk.signal_findpeaks(signal)
    >>> nk.plot_events_in_signal(signal, peaks)
    >>>
    >>> # Filter peaks
    >>> ecg = scipy.misc.electrocardiogram()
    >>> signal = ecg[0:1000]
    >>> peaks, info = nk.signal_findpeaks(signal, relative_height_min=0)
    >>> peaks2, info = nk.signal_findpeaks(signal, relative_height_min=1)
    >>> nk.plot_events_in_signal(signal, [peaks, peaks2])

    See Also
    --------
    scipy.signal.find_peaks, scipy.signal.peak_widths, peak_prominences.signal.peak_widths
    """
    peaks, info = _signal_findpeaks(signal)

    keep = np.full(len(peaks), True)

    # Absolute indices - min
    if distance_min is not None:
        keep[info["Distance"] < distance_min] = False
    if height_min is not None:
        keep[info["Height"] < height_min] = False
    if width_min is not None:
        keep[info["Width"] < width_min] = False

    # Absolute indices - max
    if distance_max is not None:
        keep[info["Distance"] > distance_max] = False
    if height_max is not None:
        keep[info["Height"] > height_max] = False
    if width_max is not None:
        keep[info["Width"] > width_max] = False

    # Relative indices - min
    if relative_distance_min is not None:
        keep[standardize(info["Distance"]) < relative_distance_min] = False
    if relative_height_min is not None:
        keep[standardize(info["Height"]) < relative_height_min] = False
    if relative_width_min is not None:
        keep[standardize(info["Width"]) < relative_width_min] = False

    # Relative indices - max
    if relative_distance_max is not None:
        keep[standardize(info["Distance"]) > relative_distance_max] = False
    if relative_height_max is not None:
        keep[standardize(info["Height"]) > relative_height_max] = False
    if relative_width_max is not None:
        keep[standardize(info["Width"]) > relative_width_max] = False

    # Filter
    peaks = peaks[keep]
    info["Distance"] = _signal_findpeaks_distances(peaks)
    info["Height"] = info["Height"][keep]
    info["Width"] = info["Width"][keep]

    return(peaks, info)
