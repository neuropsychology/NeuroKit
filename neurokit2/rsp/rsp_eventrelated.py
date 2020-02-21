# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..epochs import epochs_to_df
from ..ecg import ecg_eventrelated


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
        with each epoch indicated by the `Label` column (if not
        present, by the `Index` column). The analyzed features
        consist of the following:
        - *"RSP_Rate_Max"*: the maximum respiratory rate after stimulus onset.
        - *"RSP_Rate_Min"*: the minimum respiratory rate after stimulus onset.
        - *"RSP_Rate_Mean"*: the mean respiratory rate after stimulus onset.
        - *"RSP_Rate_Max_Time"*: the time at which maximum
        respiratory rate occurs.
        - *"RSP_Rate_Min_Time"*: the time at which minimum
        respiratory rate occurs.
        - *"RSP_Amplitude_Max"*: the maximum respiratory
        amplitude after stimulus onset.
        - *"RSP_Amplitude_Min"*: the minimum respiratory amplitude
        after stimulus onset.
        - *"RSP_Amplitude_Mean"*: the mean respiratory amplitude
        after stimulus onset.
        - *"RSP_Inspiration"*: indication of whether the onset of the event
        concurs with respiratory inspiration (1) or expiration (0).

    See Also
    --------
    events_find, epochs_create, bio_process

    Examples
    ----------
    >>> import neurokit2 as nk
    >>> import pandas as pd
    >>>
    >>> # Example with simulated data
    >>> rsp, info = nk.rsp_process(nk.rsp_simulate(duration=20))
    >>> epochs = nk.epochs_create(rsp,
                                  events=[5000, 10000, 15000],
                                  epochs_start=-0.1,
                                  epochs_end=1.9)
    >>> nk.rsp_eventrelated(epochs)
    >>>
    >>> # Example with real data
    >>> data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/example_bio_100hz.csv")
    >>>
    >>> # Process the data
    >>> df, info = nk.bio_process(rsp=data["RSP"], sampling_rate=100)
    >>> events = nk.events_find(data["Photosensor"],
                                threshold_keep='below',
                                event_conditions=["Negative",
                                                  "Neutral",
                                                  "Neutral",
                                                  "Negative"])
    >>> epochs = nk.epochs_create(df, events,
                                  sampling_rate=100,
                                  epochs_start=-0.1, epochs_end=2.9)
    >>> nk.rsp_eventrelated(epochs)
    """
    # Sanity checks
    if isinstance(epochs, pd.DataFrame):
        epochs = epochs_to_df._df_to_epochs(epochs)  # Convert df to dict

    if not isinstance(epochs, dict):
        raise ValueError("NeuroKit error: rsp_eventrelated():"
                         "Please specify an input"
                         "that is of the correct form i.e., either a dictionary"
                         "or dataframe.")

    # Warning for long epochs
    for i in epochs:
        if (np.max(epochs[i].index.values) > 10):
            print("Neurokit warning: rsp_eventrelated():"
                  "Epoch length is too long. You might want to use"
                  "rsp_periodrelated().")

    # Extract features and build dataframe
    rsp_df = {}  # Initialize an empty dict
    for epoch_index in epochs:

        rsp_df[epoch_index] = {}  # Initialize empty container

        # Rate
        rsp_df[epoch_index] = _rsp_eventrelated_rate(epochs[epoch_index],
                                                     rsp_df[epoch_index])

        # Amplitude
        rsp_df[epoch_index] = _rsp_eventrelated_amplitude(epochs[epoch_index],
                                                          rsp_df[epoch_index])

        # Inspiration
        rsp_df[epoch_index] = _rsp_eventrelated_inspiration(epochs[epoch_index],
                                                            rsp_df[epoch_index])

        # Fill with more info
        rsp_df[epoch_index] = ecg_eventrelated._eventrelated_addinfo(epochs[epoch_index],
                                                    rsp_df[epoch_index])

    rsp_df = pd.DataFrame.from_dict(rsp_df, orient="index")  # Convert to a dataframe

    return rsp_df



# =============================================================================
# Internals
# =============================================================================
def _rsp_eventrelated_rate(epoch, output={}):

    # Sanitize input
    colnames = epoch.columns.values
    if len([i for i in colnames if "RSP_Rate" in i]) == 0:
        print("NeuroKit warning: rsp_eventrelated(): input does not"
              "have an `RSP_Rate` column. Will skip all rate-related features.")
        return output

    # Get baseline
    if np.min(epoch.index.values) <= 0:
        baseline = epoch["RSP_Rate"][epoch.index <= 0].values
        signal = epoch["RSP_Rate"][epoch.index > 0].values
        index = epoch.index[epoch.index > 0].values
    else:
        baseline = epoch["RSP_Rate"][np.min(epoch.index.values):np.min(epoch.index.values)].values
        signal = epoch["RSP_Rate"][epoch.index > np.min(epoch.index)].values
        index = epoch.index[epoch.index > 0].values

    # Max / Min / Mean
    output["RSP_Rate_Max"] = np.max(signal) - np.mean(baseline)
    output["RSP_Rate_Min"] = np.min(signal) - np.mean(baseline)
    output["RSP_Rate_Mean"] = np.mean(signal) - np.mean(baseline)

    # Time of Max / Min
    output["RSP_Rate_Max_Time"] = index[np.argmax(signal)]
    output["RSP_Rate_Min_Time"] = index[np.argmin(signal)]

    return output


def _rsp_eventrelated_amplitude(epoch, output={}):

    # Sanitize input
    colnames = epoch.columns.values
    if len([i for i in colnames if "RSP_Amplitude" in i]) == 0:
        print("NeuroKit warning: rsp_eventrelated(): input does not"
              "have an `RSP_Amplitude` column. Will skip all amplitude-related features.")
        return output

    # Get baseline
    if np.min(epoch.index.values) <= 0:
        baseline = epoch["RSP_Amplitude"][epoch.index <= 0].values
        signal = epoch["RSP_Amplitude"][epoch.index > 0].values
    else:
        baseline = epoch["RSP_Amplitude"][np.min(epoch.index.values):np.min(epoch.index.values)].values
        signal = epoch["RSP_Amplitude"][epoch.index > np.min(epoch.index)].values

    # Max / Min / Mean
    output["RSP_Amplitude_Max"] = np.max(signal) - np.mean(baseline)
    output["RSP_Amplitude_Min"] = np.min(signal) - np.mean(baseline)
    output["RSP_Amplitude_Mean"] = np.mean(signal) - np.mean(baseline)

    return output


def _rsp_eventrelated_inspiration(epoch, output={}):

    # Sanitize input
    colnames = epoch.columns.values
    if len([i for i in colnames if "RSP_Inspiration" in i]) == 0:
        print("NeuroKit warning: rsp_eventrelated(): input does not"
              "have an `RSP_Inspiration` column. Will not indicate whether"
              "event onset concurs with inspiration.")
        return output

    # Indication ofinspiration
    inspiration = epoch["RSP_Inspiration"][epoch.index > 0].iloc[0]
    output["RSP_Inspiration"] = inspiration

    return output
