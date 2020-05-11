# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..bio.analyze_utils import _eventrelated_sanitycheck
from ..ecg.ecg_eventrelated import _eventrelated_addinfo


def rsp_eventrelated(epochs, silent=False):
    """Performs event-related RSP analysis on epochs.

    Parameters
    ----------
    epochs : dict, DataFrame
        A dict containing one DataFrame per event/trial,
        usually obtained via `epochs_create()`, or a DataFrame
        containing all epochs, usually obtained via `epochs_to_df()`.
    silent : bool
        If True, silence possible warnings.

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
        - *"RSP_Phase"*: indication of whether the onset of the event
        concurs with respiratory inspiration (1) or expiration (0).
        - *"RSP_PhaseCompletion"*: indication of the stage of the current
        respiration phase (0 to 1) at the onset of the event.

    See Also
    --------
    events_find, epochs_create, bio_process

    Examples
    ----------
    >>> import neurokit2 as nk
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
    >>> data = nk.data("bio_eventrelated_100hz")
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
    epochs = _eventrelated_sanitycheck(epochs, what="rsp", silent=silent)

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
        rsp_df[epoch_index] = _eventrelated_addinfo(epochs[epoch_index], rsp_df[epoch_index])

    rsp_df = pd.DataFrame.from_dict(rsp_df, orient="index")  # Convert to a dataframe

    # Move columns to front
    colnames = rsp_df.columns.values
    if len([i for i in colnames if "Condition" in i]) == 1:
        rsp_df = rsp_df[['Condition'] + [col for col in rsp_df.columns if col != 'Condition']]
    if len([i for i in colnames if "Label" in i]) == 1:
        rsp_df = rsp_df[['Label'] + [col for col in rsp_df.columns if col != 'Label']]

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
    if len([i for i in colnames if "RSP_Phase" in i]) == 0:
        print("NeuroKit warning: rsp_eventrelated(): input does not"
              "have an `RSP_Phase` column. Will not indicate whether"
              "event onset concurs with inspiration.")
        return output

    # Indication of inspiration
    inspiration = epoch["RSP_Phase"][epoch.index > 0].iloc[0]
    output["RSP_Phase"] = inspiration
    percentage = epoch["RSP_PhaseCompletion"][epoch.index > 0].iloc[0]
    output["RSP_PhaseCompletion"] = percentage

    return output
