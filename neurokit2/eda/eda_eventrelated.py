# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..epochs import _df_to_epochs


def eda_eventrelated(epochs):
    """Performs event-related ECG analysis on epochs.

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
        for each epoch, with each epoch indicated by the Index column.
        The analyzed features consist of whether EDA activation
        occurs following the event (i.e., presence of an SCR onset) and if so,
        its corresponding peak amplitude, time of peak, rise time and recovery time.

    See Also
    --------
    events_find, epochs_create, bio_process

    Examples
    ----------
    >>> import neurokit2 as nk
    >>> import pandas as pd
    >>>
    >>> # Example with data
    >>> data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/example_bio_100hz.csv")
    >>>
    >>> # Process the data
    >>> df, info = nk.bio_process(ecg=data["ECG"],
                                          rsp=data["RSP"],
                                          eda=data["EDA"],
                                          keep=data["Photosensor"],
                                          sampling_rate=100)
    >>> events = nk.events_find(data["Photosensor"],
                                threshold_keep='below',
                                event_conditions=["Negative",
                                                  "Neutral",
                                                  "Neutral",
                                                  "Negative"])
    >>> epochs = nk.epochs_create(df, events,
                                  sampling_rate=100,
                                  epochs_duration=7, epochs_start=-0.1)
    >>> nk.eda_eventrelated(epochs)
    """
    # Sanity checks
    if isinstance(epochs, pd.DataFrame):
        epochs = _df_to_epochs(epochs)  # Convert df to dict

    if not isinstance(epochs, dict):
        raise ValueError("NeuroKit error: eda_eventrelated(): Please specify an input"
                         "that is of the correct form i.e., either a dictionary"
                         "or dataframe.")

    # Extract features and build dataframe
    eda_df = {}  # Initialize an empty dict
    for epoch_index in epochs:
        eda_df[epoch_index] = {}  # Initialize an empty dict for the current epoch
        epoch = epochs[epoch_index]

        # Sanitize input
        n = np.array(epoch.columns)
        if len([i for i, item in enumerate(n) if "EDA" in item]) == 0:
            raise ValueError("NeuroKit error: eda_eventrelated(): input does not"
                             "have any processed signals related to EDA.")

        # Detect activity following the event
        activations = len(np.where(epoch["SCR_Onsets"][epoch.index > 0] != 0))
        if any(epoch["SCR_Onsets"][epoch.index > 0] != 0):
            eda_df[epoch_index]["Presence_of_Activation"] = activations
        else:
            eda_df[epoch_index]["Presence_of_Activation"] = 0

        # Analyze based on if activations are present
        if (eda_df[epoch_index]["Presence_of_Activation"] != 0):

            # Calculate amplitude and time of peak
            first_activation = np.where(epoch["SCR_Amplitude"][epoch.index > 0] != 0)[0][0]
            peak_amplitude = epoch["SCR_Amplitude"][epoch.index > 0].iloc[first_activation]
            eda_df[epoch_index]["Peak_Amplitude"] = peak_amplitude
            eda_df[epoch_index]["Time_of_Peak"] = epoch["SCR_Amplitude"][epoch.index > 0].index[first_activation]

            # Calculate rise time
            rise_time = epoch["SCR_RiseTime"][epoch.index > 0].iloc[first_activation]
            eda_df[epoch_index]["Rise_Time"] = rise_time

            # Calculate recovery time
            if any(epoch["SCR_RecoveryTime"][epoch.index > 0] != 0):
                recovery_time = np.where(epoch["SCR_RecoveryTime"][epoch.index > 0] != 0)[0][0]
                eda_df[epoch_index]["Recovery_Time"] = recovery_time
            else:
                eda_df[epoch_index]["Recovery_Time"] = "NA"

        # Otherwise NA elsewhere
        else:
            eda_df[epoch_index]["Peak_Amplitude"] = "NA"
            eda_df[epoch_index]["Time_of_Peak"] = "NA"
            eda_df[epoch_index]["Rise_Time"] = "NA"
            eda_df[epoch_index]["Recovery_Time"] = "NA"

    eda_df = pd.DataFrame.from_dict(eda_df, orient="index")  # Convert to a dataframe

    return eda_df
