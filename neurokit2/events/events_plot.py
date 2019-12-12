# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import matplotlib.cm
import matplotlib.pyplot as plt



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
            color_map = matplotlib.cm.get_cmap('rainbow')
            color = color_map(np.linspace(0, 1, num=len(events)))
        if isinstance(linestyle, str):
            linestyle = np.full(len(events), linestyle)

        # Loop through sublists
        for i, event in enumerate(events):
            for j in events[i]:
                plt.axvline(j, color=color[i], linestyle=linestyle[i])





def events_plot(events, signal=None, show=True, color="red", linestyle="--"):
    """
    Plot events in signal.

    Parameters
    ----------
    signal : array or DataFrame
        Signal array (can be a dataframe with many signals).
    events : list, ndarray or dict
        Events onset location. Can also be a list of lists, in which case it will mark them with different colors. If a dict is passed (e.g., from 'events_find()'), will select only the 'Onset' list.
    show : bool
        If True, will return a plot. If False, will return a DataFrame that can be plotted externally.
    color, linestyle : str
        Other arguments to pass to matplotlib plotting.

    See Also
    --------
    events_find

    Examples
    ----------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> nk.events_plot([1, 3, 5])
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    >>> events = nk.events_find(signal)
    >>> nk.events_plot(events, signal)
    >>>
    >>> events1 = events["Onset"]
    >>> events2 = np.arange(0, 1000, 100)
    >>> nk.events_plot([events1, events2], signal)
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=70, num=1000))
    >>> events = nk.events_find(signal)
    >>> events = [[i] for i in events['Onset']]
    >>> nk.events_plot(events, signal)
    """

    if isinstance(events, dict):
        events = events["Onset"]

    if signal is None:
        signal = np.full(events[-1]+1, 0)
    if isinstance(signal, pd.DataFrame) is False:
        signal = pd.DataFrame({"Signal": signal})


    # Plot if necessary
    if show:
        signal.plot()
        _events_plot(events, color=color, linestyle=linestyle)

    else:
        signal["Event_Onset"] = 0
        signal.iloc[events] = 1
        return(signal)



