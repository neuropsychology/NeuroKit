# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..events.events_plot import _events_plot

def signal_plot(signal):
    """
    Plot signal with events as vertical lines.

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
    if isinstance(signal, pd.DataFrame) is False:
        signal = pd.DataFrame({"Signal": signal})

    events = []
    for col in signal.columns:
        vector = signal[col]
        if vector.nunique() == 2:
            indices = np.where(vector == np.max(vector.unique()))
            if bool(np.any(np.diff(indices)==1)) is False:
                events.append(indices[0])
            else:
                vector.plot()
        else:
            vector.plot()

    if len(events) > 0:
        _events_plot(events)
