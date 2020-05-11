# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..epochs.epochs_to_df import _df_to_epochs



def _eventrelated_sanitycheck(epochs, what="ecg", silent=False):
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
