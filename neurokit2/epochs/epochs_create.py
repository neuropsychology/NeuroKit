# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..events.events_find import _events_find_label
from ..misc import listify


def epochs_create(data, events, sampling_rate=1000, epochs_duration=1, epochs_start=0, event_labels=None, event_conditions=None, baseline_correction=False):
    """
    Epoching a dataframe.

    Parameters
    ----------
    data : DataFrame
        A DataFrame containing the different signal(s) as different columns. If a vector of values is passed, it will be transformed in a DataFrame with a single 'Signal' column.
    events : list, ndarray or dict
        Events onset location. If a dict is passed (e.g., from 'events_find()'), will select only the 'onset' list.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    epochs_duration : int or list
        Duration(s) of each epochs (in seconds).
    epochs_start : int
        Epochs start relative to events_onsets (in seconds). Can be negative to start epochs before a given event.
    event_labels : list
        A list containing unique event identifiers. If `None`, will use the event index number.
    event_conditions : list
        An optional list containing, for each event, for example the trial category, group or experimental conditions.
    baseline_correction :


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
    >>> import pandas as pd
    >>>
    >>> # Get data
    >>> data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/example_bio_100hz.csv")
    >>>
    >>> # Find events
    >>> events = nk.events_find(data["Photosensor"], threshold_keep='below', event_conditions=["Negative", "Neutral", "Neutral", "Negative"])
    >>> nk.events_plot(events, data)
    >>>
    >>> # Create epochs
    >>> epochs = nk.epochs_create(data, events, sampling_rate=200, epochs_duration=3)
    >>> nk.epochs_plot(epochs)
    >>>
    >>> # Baseline correction
    >>> epochs = nk.epochs_create(data, events, sampling_rate=200, epochs_duration=3, baseline_correction=True)
    >>> nk.epochs_plot(epochs)
    """
    # Sanitize events input
    if isinstance(events, dict) is False:
        events = _events_find_label({"onset": events}, event_labels=event_labels, event_conditions=event_conditions)

    event_onsets = list(events["onset"])
    event_labels = list(events["label"])
    if 'condition' in events.keys():
        event_conditions = list(events["condition"])


    # Santize data input
    if isinstance(data, list) or isinstance(data, np.ndarray) or isinstance(data, pd.Series):
        data = pd.DataFrame({"Signal": list(data)})


    # Create epochs
    parameters = listify(onset=event_onsets, label=event_labels, condition=event_conditions, start=epochs_start, duration=epochs_duration)

    # Find the maximum numbers in an epoch
    # Then extend data by the max samples in epochs * NaN
    epoch_max_duration = int(max((i * sampling_rate
                                  for i in parameters["duration"])))
    buffer = pd.DataFrame(index=range(epoch_max_duration), columns=data.columns)
    data = data.append(buffer, ignore_index=True, sort=False)
    data = buffer.append(data, ignore_index=True, sort=False)

    # Adjust the Onset of the events
    parameters["onset"] = [i + epoch_max_duration for i in parameters["onset"]]

    epochs = {}
    for i, label in enumerate(parameters["label"]):

        # Find indices
        start = parameters["onset"][i] + (parameters["start"][i] * sampling_rate)
        end = start + (parameters["duration"][i] * sampling_rate)

        # Slice dataframe
        epoch = data.iloc[int(start):int(end)].copy()

        # Correct index
        epoch["Index"] = epoch.index.values
        epoch.index = np.linspace(start=parameters["start"][i], stop=parameters["start"][i] + parameters["duration"][i], num=len(epoch), endpoint=True)

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
