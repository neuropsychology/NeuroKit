# -*- coding: utf-8 -*-
import itertools

from warnings import warn

import numpy as np
import pandas as pd

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
    """**Find Events**

    Find and select events in a continuous signal (e.g., from a photosensor).

    Parameters
    ----------
    event_channel : array or list or DataFrame
        The channel containing the events. If multiple channels are entered, the channels are
        handled as reflecting different events new channel is created based on them.
    threshold : str or float
        The threshold value by which to select the events. If ``"auto"``, takes the value between
        the max and the min. If ``"auto"`` is used with multi-channel inputs, the default value
        of 0.9 is used to capture all events.
    threshold_keep : str
        ``"above"`` or ``"below"``, define the events as above or under the threshold. For
        photosensors, a white screen corresponds usually to higher values. Therefore, if your
        events are signaled by a black colour, events values are the lower ones (i.e., the signal
        "drops" when the events onset), and you should set the cut to ``"below"``.
    start_at : int
        Keep events which onset is after a particular time point.
    end_at : int
        Keep events which onset is before a particular time point.
    duration_min : int
        The minimum duration of an event to be considered as such (in time points).
    duration_max : int
        The maximum duration of an event to be considered as such (in time points).
    inter_min : int
        The minimum duration after an event for the subsequent event to be considered as such (in
        time points). Useful when spurious consecutive events are created due to very high sampling
        rate.
    discard_first : int
        Discard first or last n events. Useful if the experiment starts with some spurious events.
        If ``discard_first=0``, no first event is removed.
    discard_last : int
        Discard first or last n events. Useful if the experiment ends with some spurious events.
        If ``discard_last=0``, no last event is removed.
    event_labels : list
        A list containing unique event identifiers. If ``None``, will use the event index number.
    event_conditions : list
        An optional list containing, for each event, for example the trial category, group or
        experimental conditions. This option is ignored when multiple channels are supplied, as
        the function generates these automatically.

    Returns
    ----------
    dict
        Dict containing 3 to 5 arrays, ``"onset"`` for event onsets, ``"duration"`` for event
        durations, ``"label"`` for the event identifiers, the optional ``"condition"`` passed
        to ``event_conditions`` and the ``events_channel`` if multiple channels were supplied
        to the function.

    See Also
    --------
    events_plot, events_to_mne, events_create

    Example
    ----------
    Simulate a trigger signal (e.g., from photosensor)

    .. ipython:: python

      import neurokit2 as nk
      import numpy as np

      signal = np.zeros(200)
      signal[20:60] = 1
      signal[100:105] = 1
      signal[130:170] = 1

      events = nk.events_find(signal)
      events

      @savefig p_events_find1.png scale=100%
      nk.events_plot(events, signal)
      @suppress
      plt.close()

    The second event is an artifact (too short), we can skip it

    .. ipython:: python

      events = nk.events_find(signal, duration_min=10)

      @savefig p_events_find2.png scale=100%
      nk.events_plot(events, signal)
      @suppress
      plt.close()

    Combine multiple digital signals into a single channel and its compute its events
    The higher the channel, the higher the bit representation on the channel.

    .. ipython:: python

      signal2 = np.zeros(200)
      signal2[65:80] = 1
      signal2[110:125] = 1
      signal2[175:190] = 1

      @savefig p_events_find3.png scale=100%
      nk.signal_plot([signal, signal2])
      @suppress
      plt.close()

      events = nk.events_find([signal, signal2])
      events
      @savefig p_events_find4.png scale=100%
      nk.events_plot(events, events["events_channel"])
      @suppress
      plt.close()

    Convert the event condition results its human readable representation

    .. ipython:: python

      value_to_condition = {1: "Stimulus 1", 2: "Stimulus 2", 3: "Stimulus 3"}
      events["condition"] = [value_to_condition[id] for id in events["condition"]]
      events

    """
    events = _events_find(
        event_channel, threshold=threshold, threshold_keep=threshold_keep
    )

    # Warning when no events detected
    if len(events["onset"]) == 0:
        warn(
            "No events found. Check your event_channel or adjust 'threshold' or 'keep' arguments.",
            category=NeuroKitWarning,
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
        events["duration"] = events["duration"][events["onset"] >= start_at]
        events["onset"] = events["onset"][events["onset"] >= start_at]

    if end_at is not None:
        events["duration"] = events["duration"][events["onset"] <= end_at]
        events["onset"] = events["onset"][events["onset"] <= end_at]

    # Remove based on interval min
    if inter_min > 0:
        inter = np.diff(events["onset"])
        events["onset"] = np.concatenate(
            [events["onset"][0:1], events["onset"][1::][inter >= inter_min]]
        )
        events["duration"] = np.concatenate(
            [events["duration"][0:1], events["duration"][1::][inter >= inter_min]]
        )

    # Remove first and last n
    if discard_first > 0:
        events["onset"] = events["onset"][discard_first:]
        events["duration"] = events["duration"][discard_first:]
    if discard_last > 0:
        events["onset"] = events["onset"][0 : -1 * discard_last]
        events["duration"] = events["duration"][0 : -1 * discard_last]

    events = _events_find_label(
        events, event_labels=event_labels, event_conditions=event_conditions
    )

    return events


# =============================================================================
# Internals
# =============================================================================


def _events_find_label(
    events, event_labels=None, event_conditions=None, function_name="events_find"
):
    # Get n events
    n = len(events["onset"])

    # Labels
    if event_labels is None:
        event_labels = (np.arange(n) + 1).astype(str)

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
            + str(len(event_labels))
            + " `event_labels`, but "
            + str(n)
            + " events got detected :(. Check your event names or the event signal!"
        )

    events["label"] = event_labels

    # Condition
    if event_conditions is not None and "condition" not in events:
        if len(event_conditions) != n:
            raise ValueError(
                "NeuroKit error: "
                + function_name
                + "(): oops, it seems like you provided "
                + str(len(event_conditions))
                + " `event_conditions`, but "
                + str(n)
                + " events got detected :(. Check your event conditions or the event signal!"
            )
        events["condition"] = event_conditions
    return events


def _events_find(event_channel, threshold="auto", threshold_keep="above"):
    events_channel = _events_generate_events_channel(event_channel)

    # Differing setup based on multi-channel input or single channel input
    if events_channel is not None:
        if threshold == "auto":
            threshold = 0.9
        binary = signal_binarize(events_channel, threshold=threshold)
    else:
        binary = signal_binarize(event_channel, threshold=threshold)

    if threshold_keep not in ["above", "below"]:
        raise ValueError(
            "In events_find(), 'threshold_keep' must be one of 'above' or 'below'."
        )

    if threshold_keep != "above":
        binary = np.abs(binary - 1)  # Reverse if events are below

    # Initialize data
    events = {"onset": [], "duration": []}

    if events_channel is not None:
        events["events_channel"] = events_channel
        events["condition"] = []

    index = 0
    for event, group in itertools.groupby(binary):
        duration = len(list(group))
        if event == 1:
            events["onset"].append(index)
            events["duration"].append(duration)

            if events_channel is not None:
                events["condition"].append(int(events["events_channel"][index]))
        index += duration

    # Convert to array
    events["onset"] = np.array(events["onset"])
    events["duration"] = np.array(events["duration"])
    return events


def _events_generate_events_channel(event_channels):
    # check if nested list / array
    is_nested_loop = isinstance(event_channels, (list, np.ndarray)) and (
        len(event_channels) > 1 and isinstance(event_channels[0], (list, np.ndarray))
    )

    # check if dataframe
    is_dataframe = (
        isinstance(event_channels, pd.DataFrame) and len(event_channels.columns) > 1
    )

    # if neither, return None and continue
    if not is_nested_loop and not is_dataframe:
        return None

    stim_channel = None

    # create stim events array
    if is_dataframe:
        stim_channel = np.zeros(event_channels.shape[0])

        # add channels based on order and multiply by power of 2
        for i, column in enumerate(event_channels):
            peak_value = np.max(event_channels[column])
            stim_channel += np.floor(event_channels[column] / peak_value) * 2**i

    elif is_nested_loop:
        stim_channel = np.zeros(len(event_channels[0]))
        for i, channel in enumerate(event_channels):
            peak_value = np.max(channel)
            stim_channel += np.floor(channel / peak_value) * 2**i

    return stim_channel
