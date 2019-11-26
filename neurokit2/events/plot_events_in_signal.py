# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#import matplotlib.pyplot as plt
#
#
#
#
#
#
def plot_events_in_signal(signal, events, show=True, color="red", linestyle="--"):
    """
    Plot events in signal.

    Parameters
    ----------
    signal : array or DataFrame
        Signal array (can be a dataframe with many signals).
    events_onsets : list or ndarray
        Events onset location. If a dict is passed (e.g., from 'events_find()'), will select only the 'Onset' list.
    show : bool
        If True, will return a plot. If False, will return a DataFrame that can be plotted externally.
    color, linestyle : str
        Other arguments to pass to matplotlib plotting.

    Examples
    ----------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    >>> events = nk.events_find(signal)
    >>> nk.plot_events_in_signal(signal, events)
    """

    if isinstance(events, dict):
        events = events["Onset"]
#
#    if isinstance(signal, pd.DataFrame) is False:
#        df = pd.DataFrame({"Signal": signal})
#
#
#    # Plot if necessary
#    if show:
#        df.plot()
#        for event in events:
#            plt.axvline(event, color=color, linestyle=linestyle)
#
#    else:
#        df["Event_Onset"] = 0
#        df.iloc[events] = 1
#        return(df)
#


