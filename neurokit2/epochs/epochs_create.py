# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..events.events_find import _events_find_label
from ..misc import listify


def epochs_create(
    data,
    events=None,
    sampling_rate=1000,
    epochs_start=0,
    epochs_end=1,
    event_labels=None,
    event_conditions=None,
    baseline_correction=False,
):
    """
    Epoching a dataframe.

    Parameters
    ----------
    data : DataFrame
        A DataFrame containing the different signal(s) as different columns.
        If a vector of values is passed, it will be transformed in a DataFrame
        with a single 'Signal' column.
    events : list, ndarray or dict
        Events onset location. If a dict is passed (e.g., from ``events_find()``),
        will select only the 'onset' list. If an integer is passed,
        will use this number to create an evenly spaced list of events. If None,
        will chunk the signal into successive blocks of the set duration.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    epochs_start, epochs_end : int
        Epochs start and end relative to events_onsets (in seconds). The start can be negative to start epochs before a given event (to have a baseline for instance).
    event_labels : list
        A list containing unique event identifiers. If `None`, will use the event index number.
    event_conditions : list
        An optional list containing, for each event, for example the trial category, group or experimental conditions.
    baseline_correction : bool


    Returns
    ----------
    dict
        A dict containing DataFrames for all epochs.


    See Also
    ----------
    events_find, events_plot, epochs_to_df, epochs_plot

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Get data
    >>> data = nk.data("bio_eventrelated_100hz")
    >>>
    >>> # Find events
    >>> events = nk.events_find(data["Photosensor"], threshold_keep='below', event_conditions=["Negative", "Neutral", "Neutral", "Negative"])
    >>> fig1 = nk.events_plot(events, data)
    >>> fig1 #doctest: +SKIP
    >>>
    >>> # Create epochs
    >>> epochs = nk.epochs_create(data, events, sampling_rate=100, epochs_end=3)
    >>> fig2 = nk.epochs_plot(epochs)
    >>> fig2 #doctest: +SKIP
    >>>
    >>> # Baseline correction
    >>> epochs = nk.epochs_create(data, events, sampling_rate=100, epochs_end=3, baseline_correction=True)
    >>> fig3 = nk.epochs_plot(epochs)
    >>> fig3 #doctest: +SKIP
    >>>
    >>> # Chunk into n blocks of 1 second
    >>> epochs = nk.epochs_create(data, sampling_rate=100, epochs_end=1)

    """

    # Santize data input
    if isinstance(data, tuple):  # If a tuple of data and info is passed
        data = data[0]

    if isinstance(data, list) or isinstance(data, np.ndarray) or isinstance(data, pd.Series):
        data = pd.DataFrame({"Signal": list(data)})

    # Sanitize events input
    if events is None:
        max_duration = (np.max(epochs_end) - np.min(epochs_start)) * sampling_rate
        events = np.arange(0, len(data) - max_duration, max_duration)
    if isinstance(events, int):
        events = np.linspace(0, len(data), events + 2)[1:-1]
    if isinstance(events, dict) is False:
        events = _events_find_label({"onset": events}, event_labels=event_labels, event_conditions=event_conditions)

    event_onsets = list(events["onset"])
    event_labels = list(events["label"])
    if "condition" in events.keys():
        event_conditions = list(events["condition"])

    # Create epochs
    parameters = listify(
        onset=event_onsets, label=event_labels, condition=event_conditions, start=epochs_start, end=epochs_end
    )

    # Find the maximum numbers of samples in an epoch
    parameters["duration"] = np.array(parameters["end"]) - np.array(parameters["start"])
    epoch_max_duration = int(max((i * sampling_rate for i in parameters["duration"])))

    # Extend data by the max samples in epochs * NaN (to prevent non-complete data)
    length_buffer = epoch_max_duration
    buffer = pd.DataFrame(index=range(length_buffer), columns=data.columns)
    data = data.append(buffer, ignore_index=True, sort=False)
    data = buffer.append(data, ignore_index=True, sort=False)

    # Adjust the Onset of the events for the buffer
    parameters["onset"] = [i + length_buffer for i in parameters["onset"]]

    epochs = {}
    for i, label in enumerate(parameters["label"]):

        # Find indices
        start = parameters["onset"][i] + (parameters["start"][i] * sampling_rate)
        end = parameters["onset"][i] + (parameters["end"][i] * sampling_rate)

        # Slice dataframe
        epoch = data.iloc[int(start) : int(end)].copy()

        # Correct index
        epoch["Index"] = epoch.index.values - length_buffer
        epoch.index = np.linspace(
            start=parameters["start"][i], stop=parameters["end"][i], num=len(epoch), endpoint=True
        )

        if baseline_correction is True:
            baseline_end = 0 if epochs_start <= 0 else epochs_start
            epoch = epoch - epoch.loc[:baseline_end].mean()

        # Add additional
        epoch["Label"] = parameters["label"][i]
        if parameters["condition"][i] is not None:
            epoch["Condition"] = parameters["condition"][i]

        # Store
        epochs[label] = epoch

    return epochs
