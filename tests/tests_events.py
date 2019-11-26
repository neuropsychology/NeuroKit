import numpy as np
import pandas as pd
import neurokit2 as nk

# =============================================================================
# Events
# =============================================================================



def test_events_find():

    signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    events = nk.events_find(signal)
    assert list(events["Onset"]) == [0, 236, 550, 864]

    events = nk.events_find(signal, duration_min = 150)
    assert list(events["Onset"]) == [236, 550]

    events = nk.events_find(signal, inter_min = 300)
    assert list(events["Onset"]) == [0, 550, 864]



def test_plot_events_in_signal():

    signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    events = nk.events_find(signal)
    data = nk.plot_events_in_signal(signal, events, show=False)
    assert len(data['Event_Onset']) == 1000