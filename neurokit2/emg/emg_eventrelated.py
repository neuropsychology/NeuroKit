# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..epochs import _df_to_epochs


def emg_eventrelated(epochs):
    """Performs event-related EMG analysis on epochs.

    Parameters
    ----------
    epochs : dict, DataFrame
        A dict containing one DataFrame per event/trial,
        usually obtained via `epochs_create()`, or a DataFrame
        containing all epochs, usually obtained via `epochs_to_df()`.

    Returns
    -------
    DataFrame
        A dataframe containing the analyzed EMG features for each epoch,
        with each epoch indicated by the Index column. The analyzed features
        consist of the mean and maximum EMG amplitude (not adjusted for baseline)
        as well as whether there is muscular activation following the onset
        of the event.

    See Also
    --------
    emg_simulate, emg_process, events_find, epochs_create

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> emg = nk.emg_simulate(duration=10, sampling_rate=1000, n_bursts=3)
    >>> emg_signals, info = nk.emg_process(emg, sampling_rate=1000)
    >>> events = nk.events_find(emg_signals["EMG_Onsets"])
    >>> epochs = nk.epochs_create(emg_signals, events,
                                  sampling_rate=1000,
                                  epochs_duration=4, epochs_start=-0.1)
    >>> nk.emg_eventrelated(epochs)
    """
    # Sanity checks
    if isinstance(epochs, pd.DataFrame):
        epochs = _df_to_epochs(epochs)  # Convert df to dict

    if not isinstance(epochs, dict):
        raise ValueError("NeuroKit error: emg_eventrelated(): Please specify an input"
                         "that is of the correct form i.e., either a dictionary"
                         "or dataframe.")

    # Extract features and build dataframe
    emg_df = {}  # Initialize an empty dict
    for epoch_index in epochs:
        emg_df[epoch_index] = {}  # Initialize an empty dict for the current epoch
        epoch = epochs[epoch_index]

        # Sanitize input
        n = np.array(epoch.columns)
        if len([i for i, item in enumerate(n) if "EMG" in item]) == 0:
            raise ValueError("NeuroKit error: emg_eventrelated(): input does not"
                             "have any processed signals related to EMG.")

        emg_df[epoch_index]["Mean_EMG_Amplitude"] = epoch["EMG_Amplitude"].mean()
        emg_df[epoch_index]["Max_EMG_Amplitude"] = epoch["EMG_Amplitude"].max()
        emg_df[epoch_index]["Presence_of_Activation"] = epoch["EMG_Onsets"][epoch.index > 0].iloc[0]

    emg_df = pd.DataFrame.from_dict(emg_df, orient="index")  # Convert to a dataframe

    return emg_df
