# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from .epochs_to_df import epochs_to_df


def epochs_plot(epochs, legend=True, show=True, **kwargs):
    """Epochs visualization

    Plot epochs.

    Parameters
    ----------
    epochs : dict
        A dict containing one DataFrame per event/trial. Usually obtained via `epochs_create()`.
    legend : bool
        Display the legend (the key of each epoch).
    show : bool
        If True, will return a plot. If False, will return a DataFrame that can be plotted externally.

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
    >>>
    >>> # Example with data
    >>> data = nk.data("bio_eventrelated_100hz")
    >>> events = nk.events_find(data["Photosensor"],
    ...                         threshold_keep='below',
    ...                         event_conditions=["Negative", "Neutral", "Neutral", "Negative"])
    >>> epochs = nk.epochs_create(data, events, sampling_rate=200, epochs_end=1)
    >>> fig1 = nk.epochs_plot(epochs)
    >>> fig1 #doctest: +SKIP
    >>>
    >>> # Example with ECG Peaks
    >>> signal = nk.ecg_simulate(duration=10)
    >>> events = nk.ecg_findpeaks(signal)
    >>> epochs = nk.epochs_create(signal, events=events["ECG_R_Peaks"], epochs_start=-0.5, epochs_end=0.5)
    >>> fig2 = nk.epochs_plot(epochs)
    >>> fig2 #doctest: +SKIP

    """
    # sanitize epochs
    if isinstance(epochs, dict):
        data = epochs_to_df(epochs)

    elif isinstance(epochs, object):
        # Try loading mne
        try:
            import mne
        except ImportError as e:
            raise ImportError(
                "NeuroKit error: epochs_plot(): the 'mne' module is required for this function to run. ",
                "Please install it first (`pip install mne`).",
            ) from e

        if not isinstance(epochs, mne.Epochs):
            raise ValueError(
                "NeuroKit error: epochs_plot(): Please make sure your epochs object passed is `mne.Epochs` object. "
            )

        data = _epochs_mne_sanitize(epochs, **kwargs)

    cols = data.columns.values
    cols = [x for x in cols if x not in ["Time", "Condition", "Label", "Index"]]

    if show:
        if len(cols) == 1:
            fig, ax = plt.subplots()
            _epochs_plot(data, ax, cols[0], legend=legend)
        else:
            fig, ax = plt.subplots(nrows=len(cols))
            for i, col in enumerate(cols):
                _epochs_plot(data, ax=ax[i], col=col, legend=legend)
        return fig

    return data


def _epochs_mne_sanitize(epochs, what):
    """Channel array extraction from MNE for plotting.
    Select one or several channels by name and returns them in a dataframe.
    """

    data = epochs.to_data_frame()
    data = data.rename(columns={"time": "Time", "condition": "Condition", "epoch": "Label"})
    data["Time"] = data["Time"] / 1000  # ms to seconds

    if isinstance(what, str):
        data = data[[x for x in data.columns.values if x in ["Time", "Condition", "Label", what]]]
    # Select a few specified channels
    elif isinstance(what, list):
        data = data[[x for x in data.columns.values if x in ["Time", "Condition", "Label"] + what]]

    return data


def _epochs_plot(data, ax, col, legend):

    if "Condition" in data.columns:
        grouped = data.groupby("Condition")

        # Colors
        color_list = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
        colors = {}
        for i, cond in enumerate(set(data["Condition"])):
            colors[cond] = color_list[i]

        # Plot
        for key, group in grouped:
            df = group.pivot_table(index="Time", columns=["Condition", "Label"], values=col)
            df.plot(ax=ax, label=col, title=col, style=colors[key], legend=legend)

        # TODO: Custom legend
    else:
        data.pivot(index="Time", columns="Label", values=col).plot(
            ax=ax, label=col, title=col, legend=legend
        )
