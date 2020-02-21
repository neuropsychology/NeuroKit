# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def _df_to_epochs(data):
    # Convert dataframe of epochs created by `epochs_to_df` back into a dictionary.
    labels = data.Label.unique()
    epochs_dict = {i: pd.DataFrame for i in labels}
    for key in epochs_dict.keys():
        epochs_dict[key] = data[:][data.Label == key]
        epochs_dict[key].index = np.array(epochs_dict[key]["Time"])
        epochs_dict[key] = epochs_dict[key].drop(["Time"], axis=1)

    return epochs_dict
