# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..events import events_plot
from ..stats import standardize as nk_standardize


def signal_plot(
    signal, sampling_rate=None, subplots=False, standardize=False, labels=None, **kwargs
):
    """**Plot signal with events as vertical lines**

    Parameters
    ----------
    signal : array or DataFrame
        Signal array (can be a dataframe with many signals).
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second). Needs to be supplied if
        the data should be plotted over time in seconds. Otherwise the data is plotted over samples.
        Defaults to ``None``.
    subplots : bool
        If ``True``, each signal is plotted in a subplot.
    standardize : bool
        If ``True``, all signals will have the same scale (useful for visualisation).
    labels : str or list
        Defaults to ``None``.
    **kwargs : optional
        Arguments passed to matplotlib plotting.

    See Also
    --------
    ecg_plot, rsp_plot, ppg_plot, emg_plot, eog_plot

    Returns
    -------
    Though the function returns nothing, the figure can be retrieved and saved as follows:

    .. code-block:: console

        # To be run after signal_plot()
        fig = plt.gcf()
        fig.savefig("myfig.png")

    Examples
    ----------
    .. ipython:: python

      import numpy as np
      import pandas as pd
      import neurokit2 as nk

      signal = nk.signal_simulate(duration=10, sampling_rate=1000)
      @savefig p_signal_plot1.png scale=100%
      nk.signal_plot(signal, sampling_rate=1000, color="red")
      @suppress
      plt.close()

    .. ipython:: python

       # Simulate data
      data = pd.DataFrame({"Signal2": np.cos(np.linspace(start=0, stop=20, num=1000)),
                           "Signal3": np.sin(np.linspace(start=0, stop=20, num=1000)),
                           "Signal4": nk.signal_binarize(np.cos(np.linspace(start=0, stop=40, num=1000)))})

      # Process signal
      @savefig p_signal_plot2.png scale=100%
      nk.signal_plot(data, labels=['signal_1', 'signal_2', 'signal_3'], subplots=True)
      nk.signal_plot([signal, data], standardize=True)
      @suppress
      plt.close()

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
                if isinstance(content, pd.Series):
                    out = pd.concat(
                        [out, pd.DataFrame({content.name: content.values})],
                        axis=1,
                        sort=True,
                    )
                elif isinstance(content, pd.DataFrame):
                    out = pd.concat([out, content], axis=1, sort=True)
                else:
                    out = pd.concat(
                        [out, pd.DataFrame({"Signal" + str(i + 1): content})],
                        axis=1,
                        sort=True,
                    )
            signal = out

        # If vector is passed
        else:
            signal = pd.DataFrame({"Signal": signal})

    # Copy signal
    signal = signal.copy()

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
        signal.index = signal.index / sampling_rate
        title_x = "Time (seconds)"
    else:
        title_x = "Time"
    #        x_axis = np.linspace(0, signal.shape[0] / sampling_rate, signal.shape[0])
    #        x_axis = pd.DataFrame(x_axis, columns=["Time (s)"])
    #        signal = pd.concat([signal, x_axis], axis=1)
    #        signal = signal.set_index("Time (s)")

    # Plot accordingly
    if len(events_columns) > 0:
        events = []
        for col in events_columns:
            vector = signal[col]
            events.append(np.where(vector == np.max(vector.unique()))[0])
        events_plot(events, signal=signal[continuous_columns])
        if sampling_rate is None and pd.api.types.is_integer_dtype(signal.index):
            plt.gca().set_xlabel("Samples")
        else:
            plt.gca().set_xlabel(title_x)

    else:

        # Aesthetics
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        if len(continuous_columns) > len(colors):
            colors = plt.cm.viridis(np.linspace(0, 1, len(continuous_columns)))

        # Plot
        if standardize is True:
            signal[continuous_columns] = nk_standardize(signal[continuous_columns])

        if subplots is True:
            _, axes = plt.subplots(nrows=len(continuous_columns), ncols=1, sharex=True, **kwargs)
            for ax, col, color in zip(axes, continuous_columns, colors):
                ax.plot(signal[col], c=color, **kwargs)
        else:
            _ = signal[continuous_columns].plot(subplots=False, sharex=True, **kwargs)

        if sampling_rate is None and pd.api.types.is_integer_dtype(signal.index):
            plt.xlabel("Samples")
        else:
            plt.xlabel(title_x)

    # Tidy legend locations and add labels
    if labels is None:
        labels = continuous_columns.copy()

    if isinstance(labels, str):
        n_labels = len([labels])
        labels = [labels]
    elif isinstance(labels, list):
        n_labels = len(labels)

    if len(signal[continuous_columns].columns) != n_labels:
        raise ValueError(
            "NeuroKit error: signal_plot(): number of labels does not equal the number of plotted signals."
        )

    if subplots is False:
        plt.legend(labels, loc=1)
    else:
        for i, label in enumerate(labels):
            axes[i].legend([label], loc=1)
