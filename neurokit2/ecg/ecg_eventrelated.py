# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..epochs.eventrelated_utils import _eventrelated_sanitizeinput
from ..epochs.eventrelated_utils import _eventrelated_sanitizeoutput
from ..epochs.eventrelated_utils import _eventrelated_addinfo
from ..stats import fit_r2


def ecg_eventrelated(epochs, silent=False):
    """Performs event-related ECG analysis on epochs.

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
        A dataframe containing the analyzed ECG features
        for each epoch, with each epoch indicated by the `Label` column
        (if not present, by the `Index` column). The analyzed
        features consist of the following:
        - *"ECG_Rate_Max"*: the maximum heart rate after stimulus onset.
        - *"ECG_Rate_Min"*: the minimum heart rate after stimulus onset.
        - *"ECG_Rate_Mean"*: the mean heart rate after stimulus onset.
        - *"ECG_Rate_Max_Time"*: the time at which maximum heart rate occurs.
        - *"ECG_Rate_Min_Time"*: the time at which minimum heart rate occurs.
        - *"ECG_Atrial_Phase"*: indication of whether the onset of the event
        concurs with respiratory systole (1) or diastole (0).
        - *"ECG_Ventricular_Phase"*: indication of whether the onset of the
        event concurs with respiratory systole (1) or diastole (0).
        - *"ECG_Atrial_PhaseCompletion"*: indication of the stage of the
        current cardiac (atrial) phase (0 to 1) at the onset of the event.
         *"ECG_Ventricular_PhaseCompletion"*: indication of the stage of the
        current cardiac (ventricular) phase (0 to 1) at the onset of the event.
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
    >>> data = nk.data("bio_eventrelated_100hz")
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
    epochs = _eventrelated_sanitizeinput(epochs, what="ecg", silent=silent)

    # Extract features and build dataframe
    data = {}  # Initialize an empty dict
    for i in epochs.keys():

        data[i] = {}  # Initialize empty container

        # Rate
        data[i] = _ecg_eventrelated_rate(epochs[i], data[i])

        # Cardiac Phase
        data[i] = _ecg_eventrelated_phase(epochs[i], data[i])

        # Quality
        data[i] = _ecg_eventrelated_quality(epochs[i], data[i])

        # Fill with more info
        data[i] = _eventrelated_addinfo(epochs[i], data[i])

    df = _eventrelated_sanitizeoutput(data)

    return df


# =============================================================================
# Internals
# =============================================================================


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
    if len([i for i in colnames if "ECG_Phase_Atrial" in i]) == 0:
        print("NeuroKit warning: ecg_eventrelated(): input does not"
              "have an `ECG_Phase_Artrial` or `ECG_Phase_Ventricular` column."
              "Will not indicate whether event onset concurs with cardiac"
              "phase.")
        return output

    # Indication of atrial systole
    systole = epoch["ECG_Phase_Artrial"][epoch.index > 0].iloc[0]
    output["ECG_Phase_Artrial"] = systole
    percentage = epoch["ECG_Phase_Artrial_Completion"][epoch.index > 0].iloc[0]
    output["ECG_Phase_Artrial_Completion"] = percentage

    # Indication of ventricular systole
    systole = epoch["ECG_Phase_Ventricular"][epoch.index > 0].iloc[0]
    output["ECG_Phase_Ventricular"] = systole
    percentage = epoch["ECG_Phase_Ventricular_Completion"][epoch.index > 0].iloc[0]
    output["ECG_Phase_Ventricular_Completion"] = percentage

    return output


def _ecg_eventrelated_quality(epoch, output={}):

    # Sanitize input
    colnames = epoch.columns.values
    if len([i for i in colnames if "ECG_Quality" in i]) == 0:
        print("NeuroKit warning: ecg_eventrelated(): input does not"
              "have an `ECG_Quality` column. Quality of the signal"
              "is not computed.")
        return output

    # Average signal quality over epochs
    output["ECG_Quality_Mean"] = epoch["ECG_Quality"].mean()

    return output
