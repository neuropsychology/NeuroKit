# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from .epochs_to_df import epochs_to_df


def epochs_plot(epochs, legend=True):
    """
    Plot epochs.

    Parameters
    ----------
    epochs : dict
        A dict containing one DataFrame per event/trial. Usually obtained via `epochs_create()`.

    Returns
    ----------
    epochs : dict
        dict containing all epochs.


    See Also
    ----------
    events_find, events_plot, epochs_create, epochs_to_df

    Examples
    ----------
    >>> import neurokit2 as nk
    >>> import pandas as pd
    >>>
    >>> # Example with data
    >>> data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/example_bio_100hz.csv")
    >>> events = nk.events_find(data["Photosensor"], threshold_keep='below', event_conditions=["Negative", "Neutral", "Neutral", "Negative"])
    >>> epochs = nk.epochs_create(data, events, sampling_rate=200, epochs_duration=1)
    >>> nk.epochs_plot(epochs)
    >>>
    >>> # Example with ECG Peaks
    >>> signal = nk.ecg_simulate(duration=10)
    >>> events, info = nk.ecg_findpeaks(signal)
    >>> epochs = nk.epochs_create(signal, events=info["ECG_Peaks"], epochs_duration=1, epochs_start=-0.5)
    >>> nk.epochs_plot(epochs)
    """
    data = epochs_to_df(epochs)

    cols = data.columns.values
    cols = [x for x in cols if x not in ["Time", "Condition", "Label", "Index"]]


    if len(cols) == 1:
        fig, ax = plt.subplots()
        _epochs_plot(data, ax, cols[0], legend=True)
    else:
        fig, ax = plt.subplots(nrows=len(cols))
        for i, col in enumerate(cols):
            _epochs_plot(data, ax=ax[i], col=col, legend=True)

    return fig, ax








def _epochs_plot(data, ax, col, legend=True):

    if "Condition" in data.columns:
        grouped = data.groupby('Condition')

        # Colors
        color_list = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
        colors = {}
        for i, cond in enumerate(set(data["Condition"])):
            colors[cond] = color_list[i]

        # Plot
        for key, group in grouped:
            df = group.pivot_table(index='Time', columns=["Condition", 'Label'], values=col)
            df.plot(ax=ax, label=col, title=col, style=colors[key], legend=legend)

        # TODO: Custom legend
    else:
        data.pivot(index='Time', columns='Label', values=col).plot(ax=ax, label=col, title=col, legend=legend)
