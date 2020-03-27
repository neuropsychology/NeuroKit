# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..epochs.epochs_to_df import _df_to_epochs
from ..ecg.ecg_eventrelated import _eventrelated_addinfo


def eda_eventrelated(epochs):
    """Performs event-related EDA analysis on epochs.

    Parameters
    ----------
    epochs : dict, DataFrame
        A dict containing one DataFrame per event/trial,
        usually obtained via `epochs_create()`, or a DataFrame
        containing all epochs, usually obtained via `epochs_to_df()`.

    Returns
    -------
    DataFrame
        A dataframe containing the analyzed EDA features
        for each epoch, with each epoch indicated by the `Label` column
        (if not present, by the `Index` column). The analyzed features consist
        the following:
        - *"EDA_Activation"*: indication of whether Skin Conductance Response
        (SCR) occurs following the event (1 if an SCR onset is present and 0
        if absent) and if so, its corresponding peak amplitude,
        time of peak, rise and recovery time. If there is no occurrence of
        SCR, nans are displayed for the below features.
        - *"EDA_Peak_Amplitude"*: the peak amplitude of
        the first SCR in each epoch.
        - *"EDA_Peak_Amplitude_Time"*: the timepoint of each first SCR
        peak amplitude.
        - *"EDA_RiseTime"*: the risetime of each first SCR
        i.e., the time it takes for SCR to reach peak amplitude from onset.
        - *"EDA_RecoveryTime"*: the half-recovery time of each first SCR i.e.,
        the time it takes for SCR to decrease to half amplitude.

    See Also
    --------
    events_find, epochs_create, bio_process

    Examples
    ----------
    >>> import neurokit2 as nk
    >>> import pandas as pd
    >>>
    >>> # Example with simulated data
    >>> eda = nk.eda_simulate(duration=15, n_scr=3)
    >>> eda_signals, info = nk.eda_process(eda, sampling_rate=1000)
    >>> epochs = nk.epochs_create(eda_signals, events=[5000, 10000, 15000],
                                  sampling_rate=1000,
                                  epochs_start=-0.1, epochs_end=1.9)
    >>> nk.eda_eventrelated(epochs)
    >>>
    >>> # Example with real data
    >>> data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/dev/data/bio_eventrelated_100hz.csv")
    >>>
    >>> # Process the data
    >>> df, info = nk.bio_process(eda=data["EDA"], sampling_rate=100)
    >>> events = nk.events_find(data["Photosensor"],
                                threshold_keep='below',
                                event_conditions=["Negative",
                                                  "Neutral",
                                                  "Neutral",
                                                  "Negative"])
    >>> epochs = nk.epochs_create(df, events,
                                  sampling_rate=100,
                                  epochs_start=-0.1, epochs_end=6.9)
    >>> nk.eda_eventrelated(epochs)
    """
    # Sanity checks
    if isinstance(epochs, pd.DataFrame):
        epochs = _df_to_epochs(epochs)  # Convert df to dict

    if not isinstance(epochs, dict):
        raise ValueError("NeuroKit error: eda_eventrelated(): Please specify an input"
                         "that is of the correct form i.e., either a dictionary"
                         "or dataframe.")

    # Warning for epoch length (can be adjusted)
    for i in epochs:
        if (np.max(epochs[i].index.values) > 10):
            print("Neurokit warning: eda_eventrelated():"
                  "Epoch length is too long. You might want to use"
                  "eda_intervalrelated().")

    # Extract features and build dataframe
    eda_df = {}  # Initialize an empty dict
    for epoch_index in epochs:
        eda_df[epoch_index] = {}  # Initialize an empty dict for the current epoch
        epoch = epochs[epoch_index]

        # Detect activity following the events
        if any(epoch["SCR_Peaks"][epoch.index > 0] == 1) and any(epoch["SCR_Onsets"][epoch.index > 0] == 1):
            eda_df[epoch_index]["EDA_Activation"] = 1
        else:
            eda_df[epoch_index]["EDA_Activation"] = 0

        # Analyze based on if activations are present
        if (eda_df[epoch_index]["EDA_Activation"] != 0):
            eda_df[epoch_index] = _eda_eventrelated_features(epochs[epoch_index],
                                                             eda_df[epoch_index])
        else:
            eda_df[epoch_index]["EDA_Peak_Amplitude"] = np.nan
            eda_df[epoch_index]["EDA_Peak_Amplitude_Time"] = np.nan
            eda_df[epoch_index]["EDA_RiseTime"] = np.nan
            eda_df[epoch_index]["EDA_RecoveryTime"] = np.nan

        # Fill with more info
        eda_df[epoch_index] = _eventrelated_addinfo(epochs[epoch_index], eda_df[epoch_index])

    eda_df = pd.DataFrame.from_dict(eda_df, orient="index")  # Convert to a dataframe

    return eda_df


# =============================================================================
# Internals
# =============================================================================
def _eda_eventrelated_features(epoch, output={}):

    # Sanitize input
    colnames = epoch.columns.values
    if len([i for i in colnames if "SCR_Amplitude" in i]) == 0:
        print("NeuroKit warning: eda_eventrelated(): input does not"
              "have an `SCR_Amplitude` column. Will skip computation"
              "of SCR peak amplitude.")
        return output

    if len([i for i in colnames if "SCR_RecoveryTime" in i]) == 0:
        print("NeuroKit warning: eda_eventrelated(): input does not"
              "have an `SCR_RecoveryTime` column. Will skip computation"
              "of SCR half-recovery times.")
        return output

    if len([i for i in colnames if "SCR_RiseTime" in i]) == 0:
        print("NeuroKit warning: eda_eventrelated(): input does not"
              "have an `SCR_RiseTime` column. Will skip computation"
              "of SCR rise times.")
        return output

    # Peak amplitude and Time of peak
    first_activation = np.where(epoch["SCR_Amplitude"][epoch.index > 0] != 0)[0][0]
    peak_amplitude = epoch["SCR_Amplitude"][epoch.index > 0].iloc[first_activation]
    output["EDA_Peak_Amplitude"] = peak_amplitude
    output["EDA_Peak_Amplitude_Time"] = epoch["SCR_Amplitude"][epoch.index > 0].index[first_activation]

    # Rise Time
    rise_time = epoch["SCR_RiseTime"][epoch.index > 0].iloc[first_activation]
    output["EDA_RiseTime"] = rise_time

    # Recovery Time
    if any(epoch["SCR_RecoveryTime"][epoch.index > 0] != 0):
        recovery_time = np.where(epoch["SCR_RecoveryTime"][epoch.index > 0] != 0)[0][0]
        output["EDA_RecoveryTime"] = recovery_time
    else:
        output["EDA_RecoveryTime"] = np.nan

    return output
