import matplotlib.pyplot as plt
import numpy as np
import pytest

import neurokit2 as nk

# =============================================================================
# Events
# =============================================================================


def test_events_find():

    signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    events = nk.events_find(signal)
    assert list(events["onset"]) == [0, 236, 550, 864]

    events = nk.events_find(signal, duration_min=150)
    assert list(events["onset"]) == [236, 550]

    events = nk.events_find(signal, inter_min=300)
    assert list(events["onset"]) == [0, 550, 864]

    # No events found warning
    signal = np.zeros(1000)
    with pytest.warns(nk.misc.NeuroKitWarning, match=r'No events found.*'):
        nk.events_find(signal)


def test_events_to_mne():

    signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    events = nk.events_find(signal)
    events, event_id = nk.events_to_mne(events)
    assert event_id == {"event": 0}


def test_events_plot():

    signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    events = nk.events_find(signal)
    data = nk.events_plot(events, signal, show=False)
    assert len(data["Event_Onset"]) == 1000

    # Different events
    events1 = events["onset"]
    events2 = np.linspace(0, len(signal), 8)
    nk.events_plot([events1, events2], signal)
    fig = plt.gcf()

    for ax in fig.get_axes():
        handles, labels = ax.get_legend_handles_labels()
    assert len(handles) == len(events1) + len(events2) + 1
    assert len(labels) == len(handles)

    plt.close(fig)

    # Different conditions
    events = nk.events_find(signal, event_conditions=["A", "B", "A", "B"])
    nk.events_plot(events, signal)
    fig = plt.gcf()

    for ax in fig.get_axes():
        handles, labels = ax.get_legend_handles_labels()
    assert len(handles) == len(events) + 1
    assert len(labels) == len(handles)

    plt.close(fig)
