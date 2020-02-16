# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..epochs import _df_to_epochs


def rsp_eventrelated(epochs):
    """Performs event-related RSP analysis on epochs.

    Parameters
    ----------
    epochs : dict, DataFrame
        A dict containing one DataFrame per event/trial,
        usually obtained via `epochs_create()`, or a DataFrame
        containing all epochs, usually obtained via `epochs_to_df()`.

    Returns
    -------
    DataFrame
        A dataframe containing the analyzed RSP features for each epoch,
        with each epoch indicated by the Index column. The analyzed features
        consist of mean respiratory rate, the maximum, minimum
        and mean respiratory amplitude, all adjusted for baseline, as
        well as whether the onset of the event concurs with respiratory
        inspiration or expiration.

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
                                  epochs_duration=3, epochs_start=-0.1)
    >>> nk.rsp_eventrelated(epochs)
    """
    # Sanity checks
    if isinstance(epochs, pd.DataFrame):
        epochs = _df_to_epochs(epochs)  # Convert df to dict

    if not isinstance(epochs, dict):
        raise ValueError("NeuroKit error: rsp_eventrelated(): Please specify an input"
                         "that is of the correct form i.e., either a dictionary"
                         "or dataframe.")

    # Extract features and build dataframe
    rsp_df = {}  # Initialize an empty dict
    for epoch_index in epochs:
        rsp_df[epoch_index] = {}  # Initialize an empty dict for the current epoch
        epoch = epochs[epoch_index]

        # Sanitize input
        n = np.array(epoch.columns)
        if len([i for i, item in enumerate(n) if "RSP" in item]) == 0:
            raise ValueError("NeuroKit error: rsp_eventrelated(): input does not"
                             "have any processed signals related to RSP.")

        # If epoching starts before event
        if any(epoch.index < 0):

            rate_baseline = epoch["RSP_Rate"][epoch.index < 0].mean()
            rate_mean = epoch["RSP_Rate"][epoch.index > 0].mean()
            rsp_df[epoch_index]["Mean_RSP_Rate"] = rate_mean - rate_baseline

            amplitude_baseline_max = epoch["RSP_Amplitude"][epoch.index < 0].max()
            amplitude_max = epoch["RSP_Amplitude"][epoch.index > 0].max()
            rsp_df[epoch_index]["Max_RSP_Amplitude"] = amplitude_max - amplitude_baseline_max

            amplitude_baseline_min = epoch["RSP_Amplitude"][epoch.index < 0].min()
            amplitude_min = epoch["RSP_Amplitude"][epoch.index > 0].min()
            rsp_df[epoch_index]["Min_RSP_Amplitude"] = amplitude_min - amplitude_baseline_min

            amplitude_baseline_mean = epoch["RSP_Amplitude"][epoch.index < 0].mean()
            amplitude_mean = epoch["RSP_Amplitude"][epoch.index > 0].mean()
            rsp_df[epoch_index]["Mean_RSP_Amplitude"] = amplitude_mean - amplitude_baseline_mean

            inspiration = epoch["RSP_Inspiration"][epoch.index > 0].iloc[0]
            rsp_df[epoch_index]["RSP_Inspiration"] = inspiration
        else:
            rsp_df[epoch_index]["Mean_RSP_Rate"] = epoch["RSP_Rate"].mean()
            rsp_df[epoch_index]["Max_RSP_Amplitude"] = epoch["RSP_Amplitude"].max()
            rsp_df[epoch_index]["Min_RSP_Amplitude"] = epoch["RSP_Amplitude"].min()
            rsp_df[epoch_index]["Mean_RSP_Amplitude"] = epoch["RSP_Amplitude"].mean()
            rsp_df[epoch_index]["RSP_Inspiration"] = epoch["RSP_Inspiration"].iloc[0]

    rsp_df = pd.DataFrame.from_dict(rsp_df, orient="index")  # Convert to a dataframe

    return rsp_df
