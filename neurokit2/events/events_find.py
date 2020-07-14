# -*- coding: utf-8 -*-
import itertools
from warnings import warn

import numpy as np

from ..misc import NeuroKitWarning
from ..signal import signal_binarize


def events_find(
    event_channel,
    threshold="auto",
    threshold_keep="above",
    start_at=0,
    end_at=None,
    duration_min=1,
    duration_max=None,
    inter_min=0,
    discard_first=0,
    discard_last=0,
    event_labels=None,
    event_conditions=None,
):
    """Find and select events in a continuous signal (e.g., from a photosensor).

    Parameters
    ----------
    event_channel : array or list
        The channel containing the events.
    threshold : str or float
        The threshold value by which to select the events. If "auto", takes the value between the max
        and the min.
    threshold_keep : str
        "above" or "below", define the events as above or under the threshold. For photosensors, a
        white screen corresponds usually to higher values. Therefore, if your events are signaled by
        a black colour, events values are the lower ones, and you should set the cut to "below".
    start_at : int
        Keep events which onset is after a particular time point.
    end_at : int
        Keep events which onset is before a particular time point.
    duration_min : int
        The minimum duration of an event to be considered as such (in time points).
    duration_max : int
        The maximum duration of an event to be considered as such (in time points).
    inter_min : int
        The minimum duration after an event for the subsequent event to be considered as such (in time
        points). Useful when spurious consecutive events are created due to very high sampling rate.
    discard_first : int
        Discard first or last n events. Useful if the experiment starts with some spurious events.
        If discard_first=0, no first event is removed.
    discard_last : int
        Discard first or last n events. Useful if the experiment ends with some spurious events.
        If discard_last=0, no last event is removed.
    event_labels : list
        A list containing unique event identifiers. If `None`, will use the event index number.
    event_conditions : list
        An optional list containing, for each event, for example the trial category, group or
        experimental conditions.

    Returns
    ----------
    dict
        Dict containing 3 or 4 arrays, 'onset' for event onsets, 'duration' for event durations, 'label'
        for the event identifiers and the optional 'conditions' passed to `event_conditions`.

    See Also
    --------
    events_plot, events_to_mne

    Example
    ----------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=4)
    >>> events = nk.events_find(signal)
    >>> events #doctest: +ELLIPSIS
    {'onset': array(...),
     'duration': array(...),
     'label': array(...)}
    >>>
    >>> nk.events_plot(events, signal) #doctest: +ELLIPSIS
    <Figure ...>

    """
    events = _events_find(event_channel, threshold=threshold, threshold_keep=threshold_keep)

    # Warning when no events detected
    if len(events["onset"]) == 0:
        warn(
            "No events found. Check your event_channel or adjust 'threshold' or 'keep' arguments.",
            category=NeuroKitWarning
        )
        return events

    # Remove based on duration
    to_keep = np.full(len(events["onset"]), True)
    to_keep[events["duration"] < duration_min] = False
    if duration_max is not None:
        to_keep[events["duration"] > duration_max] = False
    events["onset"] = events["onset"][to_keep]
    events["duration"] = events["duration"][to_keep]

    # Remove based on index
    if start_at > 0:
        events["onset"] = events["onset"][events["onset"] >= start_at]
        events["duration"] = events["duration"][events["onset"] >= start_at]
    if end_at is not None:
        events["onset"] = events["onset"][events["onset"] <= end_at]
        events["duration"] = events["duration"][events["onset"] <= end_at]

    # Remove based on interval min
    if inter_min > 0:
        inter = np.diff(events["onset"])
        events["onset"] = np.concatenate([events["onset"][0:1], events["onset"][1::][inter >= inter_min]])
        events["duration"] = np.concatenate([events["duration"][0:1], events["duration"][1::][inter >= inter_min]])

    # Remove first and last n
    if discard_first > 0:
        events["onset"] = events["onset"][discard_first:]
        events["duration"] = events["duration"][discard_first:]
    if discard_last > 0:
        events["onset"] = events["onset"][0 : -1 * discard_last]
        events["duration"] = events["duration"][0 : -1 * discard_last]

    events = _events_find_label(events, event_labels=event_labels, event_conditions=event_conditions)

    return events


# =============================================================================
# Internals
# =============================================================================


def _events_find_label(events, event_labels=None, event_conditions=None, function_name="events_find"):
    # Get n events
    n = len(events["onset"])

    # Labels
    if event_labels is None:
        event_labels = (np.arange(n) + 1).astype(np.str)

    if len(list(set(event_labels))) != n:
        raise ValueError(
            "NeuroKit error: "
            + function_name
            + "(): oops, it seems like the `event_labels` that you provided "
            + "are not unique (all different). Please provide "
            + str(n)
            + " distinct labels."
        )

    if len(event_labels) != n:
        raise ValueError(
            "NeuroKit error: "
            + function_name
            + "(): oops, it seems like you provided "
            + str(n)
            + " `event_labels`, but "
            + str(n)
            + " events got detected :(. Check your event names or the event signal!"
        )

    events["label"] = event_labels

    # Condition
    if event_conditions is not None:
        if len(event_conditions) != n:
            raise ValueError(
                "NeuroKit error: "
                + function_name
                + "(): oops, it seems like you provided "
                + str(n)
                + " `event_conditions`, but "
                + str(n)
                + " events got detected :(. Check your event conditions or the event signal!"
            )
        events["condition"] = event_conditions
    return events


def _events_find(event_channel, threshold="auto", threshold_keep="above"):
    binary = signal_binarize(event_channel, threshold=threshold)

    if threshold_keep.lower() != "above":
        binary = np.abs(binary - 1)  # Reverse if events are below

    # Initialize data
    events = {"onset": [], "duration": []}

    index = 0
    for event, group in itertools.groupby(binary):
        duration = len(list(group))
        if event == 1:
            events["onset"].append(index)
            events["duration"].append(duration)
        index += duration

    # Convert to array
    events["onset"] = np.array(events["onset"])
    events["duration"] = np.array(events["duration"])
    return events
