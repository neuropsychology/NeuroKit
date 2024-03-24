import numpy as np
import pandas as pd

from .events_find import events_find


def stim_events_find(
    event_channels,
    value_to_condition=None,
    threshold=0.9,
    threshold_keep="above",
    start_at=0,
    end_at=None,
    duration_min=1,
    duration_max=None,
    inter_min=0,
    discard_first=0,
    discard_last=0,
):
    """**Find Multi-Channel Stimulus Events**

    Combine multiple digital input channels into reconstructed stimulus events (e.g., like from multiple digital inputs).
    The order of channels expresses the bit value it holds. i.e., channel 1 is bit 0, channel 2 is bit 1 etc.

    e.g., if bit 0, 1 and 3 are active, then the value becomes 2**0 + 2**1 + 2**3 = 11

    Parameters
    ----------
    event_channels : DataFrame or array or list
        The channels used for triggers.
    value_to_condition: dict
        Dictionary to convert the stimulus number into a condition label.
        e.g., stimulus 1 could be "Start Experiment"
    threshold : str or float
        The threshold value by which to select the events. For normal stimulus events, the lowest
        value is 1. However, if the events of interest are e.g., higher than 32, this value
        should be 31
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

    Returns
    ----------
    dict
        Dict containing 4 arrays, ``"onset"`` for event onsets, ``"duration"`` for event
        durations, ``"label"`` for the event identifiers and ``"conditions"`` for the value
        of the stimulus event or its representing str from the value_to_condition input.

    See Also
    --------
    events_plot, events_to_mne, events_create, events_find

    Example
    ----------
    Convert multiple digital input channels into stimulus events

    .. ipython:: python

      import neurokit2 as nk
      import numpy as np
      import pandas as pd


      channel1 = np.zeros(200)
      channel1[10:30] = 1
      channel1[60:80] = 1
      channel1[150:170] = 1

      channel2 = np.zeros(200)
      channel2[60:80] = 1
      channel2[150:170] = 1

      channel3 = np.zeros(200)
      channel3[150:170] = 1

      stim_events, stim_channel = nk.stim_events_find([channel1, channel2, channel3])
      stim_events

      @savefig p_triggers_find1.png scale=100%
      nk.events_plot(stim_events, stim_channel)
      @suppress
      plt.close()

    If we have the labels for each condition, we can include them in the triggers

    .. ipython:: python

      stim_meanings = {1: "experiment start", 3: "stimulation start", 7: "stimulation end"}
      stim_events, stim_channel  = nk.stim_events_find([channel1, channel2, channel3], stim_meanings)
      stim_events

    """
    # if the input event_channels were not a dataframe, convert them into one
    if not isinstance(event_channels, pd.DataFrame):
        channels = pd.DataFrame()
        for i, channel in enumerate(event_channels):
            channels[f"Channel {i}"] = channel

        event_channels = channels

    # create stim events array
    stim_channel = np.zeros(event_channels.shape[0])

    # add channels based on order and multiply by power of 2
    for i, column in enumerate(event_channels):
        peak_value = np.max(event_channels[column])
        stim_channel += np.floor(event_channels[column] / peak_value) * 2**i

    # find events
    events = events_find(
        stim_channel,
        threshold=threshold,
        threshold_keep=threshold_keep,
        start_at=start_at,
        end_at=end_at,
        duration_min=duration_min,
        duration_max=duration_max,
        inter_min=inter_min,
        discard_first=discard_first,
        discard_last=discard_last,
    )

    # pre-allocate list for conditions
    new_event_conditions = [None] * len(events["onset"])

    # Create array of conditions, named or with the number
    for i, event_sample in enumerate(events["onset"]):
        event_number = int(stim_channel[event_sample])

        if value_to_condition and event_number in value_to_condition:
            new_event_conditions[i] = value_to_condition[event_number]
        else:
            new_event_conditions[i] = event_number

    events["condition"] = new_event_conditions

    # return the events dictionary and the newly created stimulation channel
    return events, stim_channel
