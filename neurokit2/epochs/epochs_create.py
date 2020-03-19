# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..events.events_find import _events_find_label
from ..misc import listify


def epochs_create(data, events, signal_features=None, sampling_rate=1000, epochs_start=0, epochs_end=1, phys_event=False, event_labels=None, event_conditions=None, baseline_correction=False):
    """
    Epoching a dataframe.

    Parameters
    ----------
    data : DataFrame
        A DataFrame containing the different signal(s) as different columns.
        If a vector of values is passed, it will be transformed in a DataFrame
        with a single 'Signal' column.
    events : list, ndarray or dict
        Events onset location. If a dict is passed (e.g., from
       'events_find()'), will select only the 'onset' list.
    signal_features : DataFrame
        A DataFrame of same length as the input data in which occurences of
        additional signal features such as peaks, waves onsets, waves offsets,
        troughs or recovery marked as "1" in a list of zeros with the same
        length as input data.
        It can be the output of '_peaks' functions for the information of
        peaks or 'ecg_delineate' function for additional the information of
        ecg signal.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    epochs_start, epochs_end : int
        Epochs start and end relative to events_onsets (in seconds). The start can be negative to start epochs before a given event (to have a baseline for instance).
    phys_event : bol
        Optional, if you used '_info' as the event from which to calculate an epoch. 
        i.e. if you specify |events| as  'ecg_info['ECG_P_Peaks']', or rsp_info['RSP_Troughs']
        Default is False.
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
    >>> epochs = nk.epochs_create(data, events, sampling_rate=100, epochs_end=3)
    >>> nk.epochs_plot(epochs)
    >>>
    >>> # Baseline correction
    >>> epochs = nk.epochs_create(data, events, sampling_rate=100, epochs_end=3, baseline_correction=True)
    >>> nk.epochs_plot(epochs)
    """

    # Santize data input
    if isinstance(data, tuple):  # If a tuple of data and info is passed
        data = data[0]

    if isinstance(data, list) or isinstance(data, np.ndarray) or isinstance(data, pd.Series):
        data = pd.DataFrame({"Signal": list(data)}) 

    if signal_features is not None:
        data = pd.concat([data, signal_features], axis=1)

    # Sanitize events input
    if isinstance(events, dict) is False:
        events = _events_find_label({"onset": events}, event_labels=event_labels, event_conditions=event_conditions)
      
    event_onsets = list(events["onset"])
    event_labels = list(events["label"])
    
    if 'condition' in events.keys():
        event_conditions = list(events["condition"])

    # phys_event will take events_onset intervals, if True
    if phys_event is True
        phys_event = events["onset"][1:]-events['onsets'][:-1]
    
    


    # Create epochs
    parameters = listify(onset=event_onsets, label=event_labels, condition=event_conditions, 
                        start=epochs_start, duration=phys_event, end=epoch_end)

    # if phys_event is false, parameter['duration'] will take 
    # the specied number of seconds in epoch_start and _end (that way, the epoch duration is the cycle of the physiological measure of interest)
    if not parameters["duration"] 
        parameters["duration"] = np.array(parameters["end"]) - np.array(parameters["start"])
    
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
        end = parameters["onset"][i] + (parameters["end"][i] * sampling_rate)

        # Slice dataframe
        epoch = data.iloc[int(start):int(end)].copy()

        # Correct index
        epoch["Index"] = epoch.index.values
        epoch.index = np.linspace(start=parameters["start"][i], stop=parameters["end"][i], num=len(epoch), endpoint=True)

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