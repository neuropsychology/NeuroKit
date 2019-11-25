# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import itertools

from ..signal import signal_binarize










def _events_find(event_channel, threshold="auto", keep="above"):
    """Internal function
    """
    binary = signal_binarize(np.array(event_channel), threshold=threshold)

    if keep != 'above':
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
def events_find(event_channel, threshold="auto", keep="above", start_at=0, end_at=None, duration_min=1, duration_max=None):
    """
    Find and select events based on a continuous signal.

    Parameters
    ----------
    event_channel : array or list
        The channel containing the events.
    threshold : str or float
        The threshold value by which to select the events. If "auto", takes the value between the max and the min.
    keep : str
        "above" or "below", define the events as above or under the treshold. For photosensors, a white screen corresponds usually to higher values. Therefore, if your events are signaled by a black colour, events values are the lower ones, and you should set the cut to "below".
    start_at, end_at : int
        Keep events which onset is after, or before a particular time point.
    duration_min, duration_max : int
        The minimum or maximum duration of an event to be considered as such (in time points). Useful when spurious events are created due to very high sampling rate.

    Returns
    ----------
    dict
        Dict containing two arrays, 'onset' for events onsets and 'duration' for events durations.

    Example
    ----------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    >>> nk.events_find(signal)
    {'Onset': array([  0, 236, 550, 864]), 'Duration': array([ 79, 157, 157, 136])}

    >>> nk.plot_events_in_signal(signal, events)
    """
    events = _events_find(event_channel, threshold="auto", keep="above")

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
    events["Onset"] = events["Onset"][events["Onset"] >= start_at]
    events["Duration"] = events["Duration"][events["Onset"] >= start_at]

    if end_at is not None:
        events["Onset"] = events["Onset"][events["Onset"] <= end_at]
        events["Duration"] = events["Duration"][events["Onset"] <= end_at]


    return(events)
