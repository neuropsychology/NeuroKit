# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..events.events_plot import events_plot


def signal_plot(signal, subplots=False):
    """Plot signal with events as vertical lines.

    Parameters
    ----------
    signal : array or DataFrame
        Signal array (can be a dataframe with many signals).


    Examples
    ----------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    >>> nk.signal_plot(signal)
    >>>
    >>> data = pd.DataFrame({"Signal1": np.cos(np.linspace(start=0, stop=20, num=1000)),
                             "Signal2": np.sin(np.linspace(start=0, stop=20, num=1000)),
                             "Signal3": nk.signal_binarize(np.cos(np.linspace(start=0, stop=40, num=1000)))})
    >>> nk.signal_plot(data)
    """
    # Sanitize format
    if isinstance(signal, pd.DataFrame) is False:
        if len(np.array(signal).shape) > 1:
            signal = pd.DataFrame(np.array(signal).T)
            signal.columns = np.char.add(np.full(len(signal.columns), "Signal"), np.array(np.arange(len(signal.columns)) + 1, dtype=np.str))
        else:
            signal = pd.DataFrame({"Signal": signal})

    # Guess continuous and events columns
    continuous_columns = list(signal.columns.values)
    events_columns = []
    for col in signal.columns:
        vector = signal[col]
        if vector.nunique() == 2:
            indices = np.where(vector == np.max(vector.unique()))
            if bool(np.any(np.diff(indices) == 1)) is False:
                events_columns.append(col)
                continuous_columns.remove(col)

    # Plot accordingly
    if len(events_columns) > 0:
        events = np.array([])
        for col in events_columns:
            vector = signal[col]
            events = np.append(events, np.where(vector == np.max(vector.unique())))

        events_plot(events, signal=signal[continuous_columns])
    else:
        signal[continuous_columns].plot(subplots=subplots)
