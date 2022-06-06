import numpy as np

from .events_find import _events_find_label


def events_create(event_onsets, event_durations=None, event_labels=None, event_conditions=None):
    """**Create events dictionnary from list of onsets**

    Parameters
    ----------
    event_onsets : array or list
        A list of events onset.
    event_durations : array or list
        A list of durations. If none is passed, will take the duration
        between each onset (i.e., will assume that events are consecutive).
    event_labels : list
        A list containing unique event identifiers. If ``None``, will use the event index number.
    event_conditions : list
        An optional list containing, for each event, for example the trial category, group or
        experimental conditions.

    Returns
    ----------
    dict
        Dict containing 3 or 4 arrays, ``"onset"`` for event onsets, ``"duration"`` for event
        durations, ``"label"`` for the event identifiers and the optional ``"conditions"`` passed
        to ``event_conditions``.

    See Also
    --------
    events_plot, events_to_mne, events_find

    Example
    ----------
    .. ipython:: python

      import neurokit2 as nk

      events = nk.events_create(event_onsets = [500, 1500, 2500, 5000])
      events

      events = nk.events_create(event_onsets = [500, 1500, 2500, 5000],
                                event_labels=["S1", "S2", "S3", "S4"],
                                event_conditions=["A", "A", "B", "B"])
      events


    """
    if event_durations is None:
        event_durations = np.diff(np.concatenate(([0], event_onsets)))

    events = {"onset": event_onsets, "duration": event_durations}

    events = _events_find_label(
        events, event_labels=event_labels, event_conditions=event_conditions
    )

    return events
