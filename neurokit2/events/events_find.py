# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import itertools

from ..signal import signal_binarize










def _events_find(event_channel, threshold="auto", threshold_keep="above"):
    """Internal function
    """
    binary = signal_binarize(event_channel, threshold=threshold)

    if threshold_keep.lower() != 'above':
        binary = np.abs(binary - 1)  # Reverse if events are below

    # Initialize data
    events = {"Onset":[], "Duration":[]}

    index = 0
    for event, group in (itertools.groupby(binary)):
        duration = len(list(group))
        if event == 1:
            events["Onset"].append(index)
            events["Duration"].append(duration)
        index += duration

    # Convert to array
    events["Onset"] = np.array(events["Onset"])
    events["Duration"] = np.array(events["Duration"])
    return(events)













# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
def events_find(event_channel, threshold="auto", threshold_keep="above", start_at=0, end_at=None, duration_min=1, duration_max=None, inter_min=0, discard_first=0, discard_last=0):
    """Find and select events in a continuous signal (e.g., from a photosensor).

    Parameters
    ----------
    event_channel : array or list
        The channel containing the events.
    threshold : str or float
        The threshold value by which to select the events. If "auto", takes the value between the max and the min.
    threshold_keep : str
        "above" or "below", define the events as above or under the threshold. For photosensors, a white screen corresponds usually to higher values. Therefore, if your events are signaled by a black colour, events values are the lower ones, and you should set the cut to "below".
    start_at, end_at : int
        Keep events which onset is after, or before a particular time point.
    duration_min, duration_max : int
        The minimum or maximum duration of an event to be considered as such (in time points).
    inter_min : int
        The minimum duration after an event for the subsequent event to be considered as such (in time points). Useful when spurious consecutive events are created due to very high sampling rate.
    discard_first, discard_last : int
        Discard first or last n events. Useful if the experiment stats or ends with some spurious events. If discard_first=0 and discard_last=0, no first event or last event is removed.

    Returns
    ----------
    dict
        Dict containing two arrays, 'onset' for events onsets and 'duration' for events durations.

    See Also
    --------
    events_plot, events_to_mne

    Example
    ----------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    >>> events = nk.events_find(signal)
    >>> events
    {'Onset': array([  0, 236, 550, 864]), 'Duration': array([ 79, 157, 157, 136])}
    >>>
    >>> nk.events_plot(signal, events)
    """
    events = _events_find(event_channel, threshold=threshold, threshold_keep=threshold_keep)

    # Warning when no events detected
    if len(events["Onset"]) == 0:
        print("NeuroKit warning: events_find(): No events found. Check your event_channel or adjust 'threhsold' or 'keep' arguments.")
        return(events)

    # Remove based on duration
    to_keep = np.full(len(events["Onset"]), True)
    to_keep[events["Duration"] < duration_min] = False
    if duration_max is not None:
        to_keep[events["Duration"] > duration_max] = False
    events["Onset"] = events["Onset"][to_keep]
    events["Duration"] = events["Duration"][to_keep]

    # Remove based on index
    if start_at > 0:
        events["Onset"] = events["Onset"][events["Onset"] >= start_at]
        events["Duration"] = events["Duration"][events["Onset"] >= start_at]
    if end_at is not None:
        events["Onset"] = events["Onset"][events["Onset"] <= end_at]
        events["Duration"] = events["Duration"][events["Onset"] <= end_at]

    # Remove based on interval min
    if inter_min > 0:
        inter = np.diff(events["Onset"])
        events["Onset"] = np.concatenate([events["Onset"][0:1], events["Onset"][1::][inter >= inter_min]])
        events["Duration"] = np.concatenate([events["Duration"][0:1], events["Duration"][1::][inter >= inter_min]])

    # Remove first and last n
    if discard_first > 0:
        events["Onset"] = events["Onset"][discard_first:]
        events["Duration"] = events["Duration"][discard_first:]
    if discard_last > 0:
        events["Onset"] = events["Onset"][0:-1*discard_last]
        events["Duration"] = events["Duration"][0:-1*discard_last]

    return(events)
