# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning, find_closest
from ..stats import fit_r2
from .epochs_to_df import _df_to_epochs


def _eventrelated_sanitizeinput(epochs, what="ecg", silent=False):
    # Sanity checks
    if isinstance(epochs, pd.DataFrame):
        epochs = _df_to_epochs(epochs)  # Convert df to dict

    if not isinstance(epochs, dict):
        raise ValueError(
            "NeuroKit error: " + str(what) + "_eventrelated(): Please specify an input "
            "that is of the correct form i.e., either a dictionary "
            "or dataframe."
        )

    # Warning for long epochs
    if silent is False:
        length_mean = np.mean([np.max(epochs[i].index) - np.min(epochs[i].index) for i in epochs.keys()])
        if length_mean > 10:
            warn(
                str(what) + "_eventrelated():"
                " The duration of your epochs seems quite long. You might want"
                " to use " + str(what) + "_intervalrelated().",
                category=NeuroKitWarning
            )
    return epochs


def _eventrelated_addinfo(epoch, output={}):

    # Add label
    if "Index" in epoch.columns:
        output["Event_Onset"] = epoch.loc[np.min(np.abs(epoch.index))]["Index"]

    # Add label
    if "Label" in epoch.columns and len(set(epoch["Label"])) == 1:
        output["Label"] = epoch["Label"].values[0]

    # Add condition
    if "Condition" in epoch.columns and len(set(epoch["Condition"])) == 1:
        output["Condition"] = epoch["Condition"].values[0]

    # Add participant_id
    if "Participant" in epoch.columns and len(set(epoch["Participant"])) == 1:
        output["Participant"] = epoch["Participant"].values[0]

    return output


def _eventrelated_sanitizeoutput(data):

    df = pd.DataFrame.from_dict(data, orient="index")  # Convert to a dataframe

    colnames = df.columns.values
    if "Event_Onset" in colnames:
        df = df.sort_values("Event_Onset")
        df = df[["Event_Onset"] + [col for col in df.columns if col != "Event_Onset"]]

    # Move columns to front
    if "Condition" in colnames:
        df = df[["Condition"] + [col for col in df.columns if col != "Condition"]]
    if "Label" in colnames:
        df = df[["Label"] + [col for col in df.columns if col != "Label"]]

    return df


def _eventrelated_rate(epoch, output={}, var="ECG_Rate", time_start=None, time_end=None):

    # Sanitize input
    colnames = epoch.columns.values
    if len([i for i in colnames if var in i]) == 0:
        warn(
            "Input does not have an `" + var + "` column."
            " Will skip all rate-related features.",
            category=NeuroKitWarning
        )
        return output

    # Get baseline
    zero = find_closest(0, epoch.index.values, return_index=True)  # Find closest to 0
    baseline = epoch[var].iloc[zero]

    # Create smaller epoch within epoch provided when applicable
    # Sanitize time_start and time_end format
    if isinstance(time_start, dict):
        if var not in time_start.keys():
            time_start = None
            warn(
                "`time_start` for `" + var + "` is not provided. "
                "If it's intentional, set `time_start` for `" + var + "` to None to silent this warning.",
                category=NeuroKitWarning
                )
        else:
            time_start = time_start[var]
    if isinstance(time_end, dict):
        if var not in time_end.keys():
            time_end = None
            warn(
                "`time_end` for `" + var + "` is not provided. "
                "If it's intentional, set `time_end` for `" + var + "` to None to silent this warning.",
                category=NeuroKitWarning
                )
        else:
            time_end = time_end[var]

    # Normal behaviour if time_start and time_end are both None
    if time_start is None and time_end is None:
        signal = epoch[var].values[zero + 1 : :]
        index = epoch.index.values[zero + 1 : :]
    else:
        # Sanitize time_start and time_end values
        if time_start is None:
            time_start = epoch.index.values[zero]  # set time_start to time zero
        if time_end is None:
            time_end = epoch.index.values[-1]
        if time_start < np.min(epoch.index.values):
            time_start = epoch.index.values[0]
            warn(
                "`time_start` provided for `" + var + "` is earlier than the start of the epoch. "
                "For this analysis, `time_start` is set to the start of the epoch.",
                category=NeuroKitWarning
                )
        if time_end > np.max(epoch.index.values):
            time_end = epoch.index.values[-1]
            warn(
                "`time_end` provided for `" + var + "` is later than the end of the epoch. "
                "For this analysis, `time_end` is set to the end of the epoch.",
                category=NeuroKitWarning
                )

        # Truncate a smaller epoch within the input epoch
        epoch_start = find_closest(time_start, epoch.index.values, return_index=True)
        epoch_end = find_closest(time_end, epoch.index.values, return_index=True)
        signal = epoch[var].values[epoch_start : epoch_end + 1]
        index = epoch.index.values[epoch_start : epoch_end + 1]

    # Max / Min / Mean
    output[var + "_Baseline"] = baseline
    output[var + "_Max"] = np.max(signal) - baseline
    output[var + "_Min"] = np.min(signal) - baseline
    output[var + "_Mean"] = np.mean(signal) - baseline

    # Time of Max / Min
    output[var + "_Max_Time"] = index[np.argmax(signal)]
    output[var + "_Min_Time"] = index[np.argmin(signal)]

    # Modelling
    # These are experimental indices corresponding to parameters of a quadratic model
    # Instead of raw values (such as min, max etc.)
    coefs = np.polyfit(index, signal - baseline, 2)
    output[var + "_Trend_Quadratic"] = coefs[0]
    output[var + "_Trend_Linear"] = coefs[1]
    output[var + "_Trend_R2"] = fit_r2(
        y=signal - baseline, y_predicted=np.polyval(coefs, index), adjusted=False, n_parameters=3
    )

    return output
