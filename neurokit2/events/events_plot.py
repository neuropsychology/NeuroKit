# -*- coding: utf-8 -*-
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def events_plot(events, signal=None, show=True, color="red", linestyle="--"):
    """Plot events in signal.

    Parameters
    ----------
    events : list or ndarray or dict
        Events onset location. Can also be a list of lists, in which case it will mark them with
        different colors. If a dict is passed (e.g., from 'events_find()'), will select only the 'onset' list.
    signal : array or DataFrame
        Signal array (can be a dataframe with many signals).
    show : bool
        If True, will return a plot. If False, will return a DataFrame that can be plotted externally.
    color : str
        Argument passed to matplotlib plotting.
    linestyle : str
        Argument passed to matplotlib plotting.

    Returns
    -------
    fig
        Figure representing a plot of the signal and the event markers.

    See Also
    --------
    events_find

    Examples
    ----------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> fig = nk.events_plot([1, 3, 5])
    >>> fig #doctest: +SKIP
    >>>
    >>> # With signal
    >>> signal = nk.signal_simulate(duration=4)
    >>> events = nk.events_find(signal)
    >>> fig1 = nk.events_plot(events, signal)
    >>> fig1 #doctest: +SKIP
    >>>
    >>> # Different events
    >>> events1 = events["onset"]
    >>> events2 = np.linspace(0, len(signal), 8)
    >>> fig2 = nk.events_plot([events1, events2], signal)
    >>> fig2 #doctest: +SKIP
    >>>
    >>> # Conditions
    >>> events = nk.events_find(signal, event_conditions=["A", "B", "A", "B"])
    >>> fig3 = nk.events_plot(events, signal)
    >>> fig3 #doctest: +SKIP
    >>>
    >>> # Different colors for all events
    >>> signal = nk.signal_simulate(duration=20)
    >>> events = nk.events_find(signal)
    >>> events = [[i] for i in events['onset']]
    >>> fig4 = nk.events_plot(events, signal)
    >>> fig4 #doctest: +SKIP

    """

    if isinstance(events, dict):
        if "condition" in events.keys():
            events_list = []
            for condition in set(events["condition"]):
                events_list.append([x for x, y in zip(events["onset"], events["condition"]) if y == condition])
            events = events_list
        else:
            events = events["onset"]

    if signal is None:
        signal = np.full(events[-1] + 1, 0)
    if isinstance(signal, pd.DataFrame) is False:
        signal = pd.DataFrame({"Signal": signal})

    # Plot if necessary
    if show:
        fig = signal.plot().get_figure()
        _events_plot(events, color=color, linestyle=linestyle)
        return fig
    else:
        signal["Event_Onset"] = 0
        signal.iloc[events] = 1
        return signal


def _events_plot(events, color="red", linestyle="--"):
    # Check if events is list of lists
    try:
        len(events[0])
        is_listoflists = True
    except TypeError:
        is_listoflists = False

    if is_listoflists is False:
        # Loop through sublists
        for event in events:
            plt.axvline(event, color=color, linestyle=linestyle)

    else:
        # Convert color and style to list
        if isinstance(color, str):
            color_map = matplotlib.cm.get_cmap("rainbow")
            color = color_map(np.linspace(0, 1, num=len(events)))
        if isinstance(linestyle, str):
            linestyle = np.full(len(events), linestyle)

        # Loop through sublists
        for i, event in enumerate(events):
            for j in events[i]:
                plt.axvline(j, color=color[i], linestyle=linestyle[i], label=str(i))

        # Display only one legend per event type
        handles, labels = plt.gca().get_legend_handles_labels()
        newLabels, newHandles = [], []
        for handle, label in zip(handles, labels):
            if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)
        plt.legend(newHandles, newLabels)
