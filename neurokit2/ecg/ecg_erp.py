# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..epochs import _df_to_epochs


def ecg_erp(epochs):
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
        (i.e., heart rate variability and number of R peaks)
        for each epoch, with each epoch indicated by the Index column.

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
    >>> nk.ecg_erp(epochs)
    """
    # Sanity checks
    if isinstance(epochs, pd.DataFrame):
        epochs = _df_to_epochs(epochs)  # Convert df to dict

    if not isinstance(epochs, dict):
        raise ValueError("NeuroKit error: ecg_erp(): Please specify an input"
                         "that is of the correct form i.e., either a dictionary"
                         "or dataframe.")

    # Extract features and build dataframe
    ecg_df = {}  # Initialize an empty dict
    for epoch_index in epochs:
        ecg_df[epoch_index] = {}  # Initialize an empty dict for the current epoch
        epoch = epochs[epoch_index]

        # Sanitize input
        n = np.array(epoch.columns)
        if len([i for i, item in enumerate(n) if "ECG" in item]) == 0:
            raise ValueError("NeuroKit error: ecg_erp(): input does not"
                             "have any processed signals related to ECG.")

        if any(epoch.index < 0):
            ecg_baseline = epoch["ECG_Rate"][epoch.index < 0].mean()  # Baseline
            ecg_mean = epoch["ECG_Rate"][epoch.index > 0].mean()  # Mean heart rate when active
            ecg_df[epoch_index]["ECG_Rate"] = ecg_mean - ecg_baseline  # Correct for baseline
            ecg_max_peak = np.sum(epoch["ECG_R_Peaks"][epoch.index > 0])
            ecg_df[epoch_index]["R_Peaks"] = ecg_max_peak
        else:
            ecg_df[epoch_index]["ECG_Rate"] = epoch["ECG_Rate"].mean()
            ecg_max_peak = np.sum(epoch["ECG_R_Peaks"])
            ecg_df[epoch_index]["R_Peaks"] = ecg_max_peak

    ecg_df = pd.DataFrame.from_dict(ecg_df, orient="index")  # Convert to a dataframe

    return ecg_df
