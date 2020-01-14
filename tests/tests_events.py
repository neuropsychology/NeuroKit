import numpy as np
import pandas as pd
import neurokit2 as nk

# =============================================================================
# Events
# =============================================================================



def test_events_find():

    signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    events = nk.events_find(signal)
    assert list(events["onset"]) == [0, 236, 550, 864]

    events = nk.events_find(signal, duration_min = 150)
    assert list(events["onset"]) == [236, 550]

    events = nk.events_find(signal, inter_min = 300)
    assert list(events["onset"]) == [0, 550, 864]



def test_events_to_mne():

    signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    events = nk.events_find(signal)
    events, event_id = nk.events_to_mne(events)
    assert event_id == {'event': 0}



def test_events_plot():

    signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    events = nk.events_find(signal)
    data = nk.events_plot(events, signal, show=False)
    assert len(data['Event_Onset']) == 1000