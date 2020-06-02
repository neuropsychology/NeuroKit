# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..events import events_plot
from ..stats import standardize as nk_standardize


def signal_plot(signal, sampling_rate=None, subplots=False, standardize=False, **kwargs):
    """
    Plot signal with events as vertical lines.

    Parameters
    ----------
    signal : array or DataFrame
        Signal array (can be a dataframe with many signals).
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second). Needs
        to be supplied if the data should be plotted over time in seconds.
        Otherwise the data is plotted over samples. Defaults to None.
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
    >>> signal = nk.signal_simulate(duration=10, sampling_rate=1000)
    >>> nk.signal_plot(signal, sampling_rate=1000, color="red")
    >>>
    >>> data = pd.DataFrame({"Signal2": np.cos(np.linspace(start=0, stop=20, num=1000)), "Signal3": np.sin(np.linspace(start=0, stop=20, num=1000)), "Signal4": nk.signal_binarize(np.cos(np.linspace(start=0, stop=40, num=1000)))})
    >>> nk.signal_plot(data, subplots=True)
    >>> nk.signal_plot([signal, data], standardize=True)

    """
    # Sanitize format
    if isinstance(signal, list):
        try:
            for i in signal:
                len(i)
        except TypeError:
            signal = np.array(signal)

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

    # Adjust for sampling rate
    if sampling_rate is not None:
        x_axis = np.linspace(0, signal.shape[0] / sampling_rate, signal.shape[0])
        x_axis = pd.DataFrame(x_axis, columns=["Time (s)"])
        signal = pd.concat([signal, x_axis], axis=1)
        signal = signal.set_index("Time (s)")
    elif sampling_rate is None:
        x_axis = np.arange(0, signal.shape[0])
        x_axis = pd.DataFrame(x_axis, columns=["Samples"])
        signal = pd.concat([signal, x_axis], axis=1)
        signal = signal.set_index("Samples")

    # Plot accordingly
    if len(events_columns) > 0:
        events = []
        for col in events_columns:
            vector = signal[col]
            events.append(np.where(vector == np.max(vector.unique()))[0])
        plot = events_plot(events, signal=signal[continuous_columns])
        if sampling_rate is not None:
            plot.gca().set_xlabel("Time (seconds)")
        elif sampling_rate is None:
            plot.gca().set_xlabel("Samples")
    else:
        if standardize is True:
            plot = nk_standardize(signal[continuous_columns]).plot(subplots=subplots, sharex=True, **kwargs)
        else:
            plot = signal[continuous_columns].plot(subplots=subplots, sharex=True, **kwargs)

    # Tidy legend locations
    [plot.legend(loc=1) for plot in plt.gcf().axes]
