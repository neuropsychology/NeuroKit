# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..events import events_plot
from ..stats import standardize as nk_standardize


def signal_plot(signal, subplots=False, standardize=False):
    """Plot signal with events as vertical lines.

    Parameters
    ----------
    signal : array or DataFrame
        Signal array (can be a dataframe with many signals).
    subsubplots : bool
        If True, each signal is plotted in a subplot.
    standardize : bool
        If True, all signals will have the same scale (useful for visualisation).

    Examples
    ----------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(length=1000)
    >>> nk.signal_plot(signal)
    >>>
    >>> data = pd.DataFrame({"Signal2": np.cos(np.linspace(start=0, stop=20, num=1000)),
                             "Signal3": np.sin(np.linspace(start=0, stop=20, num=1000)),
                             "Signal4": nk.signal_binarize(np.cos(np.linspace(start=0, stop=40, num=1000)))})
    >>> nk.signal_plot(data)
    >>> nk.signal_plot([signal, data], standardize=True)
    """
    # Sanitize format
    if isinstance(signal, pd.DataFrame) is False:

        # If list is passed
        if isinstance(signal, list) or len(np.array(signal).shape) > 1:
            out = pd.DataFrame()
            for i, content in enumerate(signal):
                if isinstance(content, pd.DataFrame) or isinstance(content, pd.Series):
                    out = pd.concat([out, content], axis=1, sort=True)
                else:
                    out = pd.concat([out, pd.DataFrame({"Signal" + str(i + 1): content})], axis=1, sort=True)
            signal = out

        # If vector is passed
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
        events = []
        for col in events_columns:
            vector = signal[col]
            events.append(np.where(vector == np.max(vector.unique()))[0])

        events_plot(events, signal=signal[continuous_columns])
    else:
        if standardize is True:
            nk_standardize(signal[continuous_columns]).plot(subplots=subplots)
        else:
            signal[continuous_columns].plot(subplots=subplots)

    # Tidy legend locations
    [ax.legend(loc=1) for ax in plt.gcf().axes]
