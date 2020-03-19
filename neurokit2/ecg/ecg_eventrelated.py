# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..epochs.epochs_to_df import _df_to_epochs
from ..stats import fit_r2

def ecg_eventrelated(epochs):
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
        A dataframe containing the analyzed ECG features
        for each epoch, with each epoch indicated by the `Label` column
        (if not present, by the `Index` column). The analyzed
        features consist of the following:
        - *"ECG_Rate_Max"*: the maximum heart rate after stimulus onset.
        - *"ECG_Rate_Min"*: the minimum heart rate after stimulus onset.
        - *"ECG_Rate_Mean"*: the mean heart rate after stimulus onset.
        - *"ECG_Rate_Max_Time"*: the time at which maximum heart rate occurs.
        - *"ECG_Rate_Min_Time"*: the time at which minimum heart rate occurs.
        We also include the following *experimental* features related to the
        parameters of a quadratic model.
        - *"ECG_Rate_Trend_Linear"*: The parameter corresponding to the linear trend.
        - *"ECG_Rate_Trend_Quadratic"*: The parameter corresponding to the curvature.
        - *"ECG_Rate_Trend_R2"*: the quality of the quadratic model. If too low,
        the parameters might not be reliable or meaningful.

    See Also
    --------
    events_find, epochs_create, bio_process

    Examples
    ----------
    >>> import neurokit2 as nk
    >>> import pandas as pd
    >>>
    >>> # Example with simulated data
    >>> ecg, info = nk.ecg_process(nk.ecg_simulate(duration=20))
    >>> epochs = nk.epochs_create(ecg,
                                  events=[5000, 10000, 15000],
                                  epochs_start=-0.1,
                                  epochs_end=1.9)
    >>> nk.ecg_eventrelated(epochs)
    >>>
    >>> # Example with real data
    >>> data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/example_bio_100hz.csv")
    >>>
    >>> # Process the data
    >>> df, info = nk.bio_process(ecg=data["ECG"], sampling_rate=100)
    >>> events = nk.events_find(data["Photosensor"],
                                threshold_keep='below',
                                event_conditions=["Negative",
                                                  "Neutral",
                                                  "Neutral",
                                                  "Negative"])
    >>> epochs = nk.epochs_create(df, events,
                                  sampling_rate=100,
                                  epochs_start=-0.1, epochs_end=1.9)
    >>> nk.ecg_eventrelated(epochs)
    """
    # Sanity checks
    if isinstance(epochs, pd.DataFrame):
        epochs = _df_to_epochs(epochs)  # Convert df to dict

    if not isinstance(epochs, dict):
        raise ValueError("NeuroKit error: ecg_eventrelated():"
                         "Please specify an input"
                         "that is of the correct form i.e., either a dictionary"
                         "or dataframe as returned by `epochs_create()`.")

    # Warning for long epochs
    for i in epochs:
        if (np.max(epochs[i].index.values) > 5):
            print("Neurokit warning: ecg_eventrelated():"
                  "Epoch length is too long. You might want to use"
                  "ecg_periodrelated().")

    # Extract features and build dataframe
    ecg_df = {}  # Initialize an empty dict
    for epoch_index in epochs:

        ecg_df[epoch_index] = {}  # Initialize empty container

        # Rate
        ecg_df[epoch_index] = _ecg_eventrelated_rate(epochs[epoch_index],
                                                     ecg_df[epoch_index])

        # Cardiac Phase
        ecg_df[epoch_index] = _ecg_eventrelated_phase(epochs[epoch_index],
                                                      ecg_df[epoch_index])

        # Fill with more info
        ecg_df[epoch_index] = _eventrelated_addinfo(epochs[epoch_index],
                                                    ecg_df[epoch_index])

    ecg_df = pd.DataFrame.from_dict(ecg_df, orient="index")  # Convert to a dataframe

    return ecg_df


# =============================================================================
# Internals
# =============================================================================



def _eventrelated_addinfo(epoch, output={}):

    # Add label
    if "Label" in epoch.columns:
        if len(set(epoch["Label"])) == 1:
            output["Label"] = epoch["Label"].values[0]

    # Add condition
    if "Condition" in epoch.columns:
        if len(set(epoch["Condition"])) == 1:
            output["Condition"] = epoch["Condition"].values[0]

    # Add participant_id
    if "Participant" in epoch.columns:
        if len(set(epoch["Participant"])) == 1:
            output["Participant"] = epoch["Participant"].values[0]
    return output






def _ecg_eventrelated_rate(epoch, output={}):

    # Sanitize input
    colnames = epoch.columns.values
    if len([i for i in colnames if "ECG_Rate" in i]) == 0:
        print("NeuroKit warning: ecg_eventrelated(): input does not"
              "have an `ECG_Rate` column. Will skip all rate-related features.")
        return output

    # Get baseline
    if np.min(epoch.index.values) <= 0:
        baseline = epoch["ECG_Rate"][epoch.index <= 0].values
        signal = epoch["ECG_Rate"][epoch.index > 0].values
        index = epoch.index[epoch.index > 0].values
    else:
        baseline = epoch["ECG_Rate"][np.min(epoch.index.values):np.min(epoch.index.values)].values
        signal = epoch["ECG_Rate"][epoch.index > np.min(epoch.index)].values
        index = epoch.index[epoch.index > 0].values

    # Max / Min / Mean
    output["ECG_Rate_Max"] = np.max(signal) - np.mean(baseline)
    output["ECG_Rate_Min"] = np.min(signal) - np.mean(baseline)
    output["ECG_Rate_Mean"] = np.mean(signal) - np.mean(baseline)

    # Time of Max / Min
    output["ECG_Rate_Max_Time"] = index[np.argmax(signal)]
    output["ECG_Rate_Min_Time"] = index[np.argmin(signal)]

    # Modelling
    # These are experimental indices corresponding to parameters of a quadratic model
    # Instead of raw values (such as min, max etc.)
    coefs = np.polyfit(index, signal - np.mean(baseline), 2)
    output["ECG_Rate_Trend_Quadratic"] = coefs[0]
    output["ECG_Rate_Trend_Linear"] = coefs[1]
    output["ECG_Rate_Trend_R2"] = fit_r2(
            y=signal - np.mean(baseline),
            y_predicted=np.polyval(coefs, index),
            adjusted=False,
            n_parameters=3)

    return output


def _ecg_eventrelated_phase(epoch, output={}):

    # Sanitize input
    colnames = epoch.columns.values
    if len([i for i in colnames if "ECG_Atrial_Phase" in i]) == 0:
        print("NeuroKit warning: ecg_eventrelated(): input does not"
              "have an `ECG_Atrial_Phase` or `ECG_Ventricular_Phase` column."
              "Will not indicate whether event onset concurs with cardiac"
              "phase.")
        return output

    # Indication of atrial systole
    systole = epoch["ECG_Atrial_Phase"][epoch.index > 0].iloc[0]
    output["ECG_Atrial_Phase"] = systole
    percentage = epoch["ECG_Atrial_PhaseCompletion"][epoch.index > 0].iloc[0]
    output["ECG_Atrial_PhaseCompletion"] = percentage

    # Indication of ventricular systole
    systole = epoch["ECG_Ventricular_Phase"][epoch.index > 0].iloc[0]
    output["ECG_Ventricular_Phase"] = systole
    percentage = epoch["ECG_Ventricular_PhaseCompletion"][epoch.index > 0].iloc[0]
    output["ECG_Ventricular_PhaseCompletion"] = percentage

    return output
