# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from .epochs_to_df import _df_to_epochs





def _eventrelated_sanitizeinput(epochs, what="ecg", silent=False):
    # Sanity checks
    if isinstance(epochs, pd.DataFrame):
        epochs = _df_to_epochs(epochs)  # Convert df to dict

    if not isinstance(epochs, dict):
        raise ValueError("NeuroKit error: " + str(what) + "_eventrelated(): Please specify an input"
                         "that is of the correct form i.e., either a dictionary"
                         "or dataframe.")

    # Warning for long epochs
    if silent is False:
        length_mean = np.mean([np.max(epochs[i].index) - np.min(epochs[i].index)  for i in epochs.keys()])
        if length_mean > 10:
            print("Neurokit warning: " + str(what) + "_eventrelated():"
                  "The duration of your epochs seems quite long. You might want"
                   " to use " + str(what) + "_intervalrelated().")
    return epochs




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





def _eventrelated_sanitizeoutput(data):

    df = pd.DataFrame.from_dict(data, orient="index")  # Convert to a dataframe

    # Move columns to front
    colnames = df.columns.values
    if len([i for i in colnames if "Condition" in i]) == 1:
        df = df[['Condition'] + [col for col in df.columns if col != 'Condition']]
    if len([i for i in colnames if "Label" in i]) == 1:
        df = df[['Label'] + [col for col in df.columns if col != 'Label']]

    return df
