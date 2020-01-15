# -*- coding: utf-8 -*-
import numpy as np






def events_to_mne(events, conditions=None):
    """Create `MNE <https://mne.tools/stable/index.html>`_ compatible events for integration with M/EEG..

    Parameters
    ----------
    events : list, ndarray or dict
        Events onset location. If a dict is passed (e.g., from 'events_find()'), will select only the 'Onset' list.
    conditions : list
        A list of same length as events containing the stimuli types/conditions.


    Returns
    ----------
    (events, event_id) : tuple
        MNE-formatted events, that can be added via 'raw.add_events(events), and a dictionary with event's names.

    See Also
    --------
    events_find

    Examples
    ----------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    >>> events = nk.events_find(signal)
    >>> events, event_id = nk.events_to_mne(events)
    >>> event_id
    {'Event': 0}
    """

    if isinstance(events, dict):
        events = events["Onset"]

    event_id = {}

    if conditions is None:
        conditions = ["Event"] * len(events)

    # Sanity check
    if len(conditions) != len(events):
        raise ValueError("NeuroKit error: events_to_mne(): 'conditions' argument of different length than event onsets.")


    event_names = list(set(conditions))
    event_index = list(range(len(event_names)))
    for i in enumerate(event_names):
        conditions = [event_index[i[0]] if x==i[1] else x for x in conditions]
        event_id[i[1]] = event_index[i[0]]

    events = np.array([events, [0]*len(events), conditions]).T
    return(events, event_id)
