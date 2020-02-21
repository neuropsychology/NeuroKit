# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..epochs import epochs_to_df
from ..ecg import ecg_eventrelated


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
        with each epoch indicated by the `Label` column (if not present,
        by the `Index` column). The analyzed features consist of the following:
        - *"EMG_Amplitude_Max"*: the maximum amplitude of the activity.
        - *"EMG_Amplitude_Mean"*: the mean amplitude of the activity.
        - *"EMG_Activation"*: indication of whether there is muscular activation
        following the onset of the event (1 if present, 0 if absent).

    See Also
    --------
    emg_simulate, emg_process, events_find, epochs_create

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Example with simulated data
    >>> emg = nk.emg_simulate(duration=20, sampling_rate=1000, n_bursts=3)
    >>> emg_signals, info = nk.emg_process(emg, sampling_rate=1000)
    >>> epochs = nk.epochs_create(emg_signals, events=[5000, 10000, 15000],
                                  sampling_rate=1000,
                                  epochs_start=-0.1, epochs_end=1.9)
    >>> nk.emg_eventrelated(epochs)
    """
    # Sanity checks
    if isinstance(epochs, pd.DataFrame):
        epochs = epochs_to_df._df_to_epochs(epochs)  # Convert df to dict

    if not isinstance(epochs, dict):
        raise ValueError("NeuroKit error: emg_eventrelated(): Please specify an input"
                         "that is of the correct form i.e., either a dictionary"
                         "or dataframe.")

    # Warning for long epochs
    for i in epochs:
        if (np.max(epochs[i].index.values) > 5):
            print("Neurokit warning: emg_eventrelated():"
                  "Epoch length is too long. You might want to use"
                  "emg_periodrelated().")

    # Extract features and build dataframe
    emg_df = {}  # Initialize an empty dict
    for epoch_index in epochs:
        emg_df[epoch_index] = {}  # Initialize an empty dict for the current epoch
        epoch = epochs[epoch_index]

        # Sanitize input
        n = np.array(epoch.columns)
        if len([i for i, item in enumerate(n) if "EMG_Amplitude" in item]) == 0:
            raise ValueError("NeuroKit error: emg_eventrelated(): input does not"
                             "have an `EMG_Amplitude` column. Will skip all"
                             "amplitude-related features.")
        if len([i for i, item in enumerate(n) if "EMG_Onsets" in item]) == 0:
            raise ValueError("NeuroKit error: emg_eventrelated(): input does not"
                             "have an `EMG_Onsets` column. Will not indicate"
                             "whether muscular activation follows event onset.")

        # Amplitude
        emg_df[epoch_index]["EMG_Amplitude_Mean"] = epoch["EMG_Amplitude"].mean()
        emg_df[epoch_index]["EMG_Amplitude_Max"] = epoch["EMG_Amplitude"].max()

        # Activation following event
        activations = len(np.where(epoch["EMG_Onsets"][epoch.index > 0] != 0))
        if any(epoch["EMG_Onsets"][epoch.index > 0] != 0):
            emg_df[epoch_index]["EMG_Activation"] = activations
        else:
            emg_df[epoch_index]["EMG_Activation"] = 0

        # Fill with more info
        emg_df[epoch_index] = ecg_eventrelated._eventrelated_addinfo(epoch, emg_df[epoch_index])

    emg_df = pd.DataFrame.from_dict(emg_df, orient="index")  # Convert to a dataframe

    return emg_df
