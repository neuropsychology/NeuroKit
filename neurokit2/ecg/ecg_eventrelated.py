# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..epochs import _df_to_epochs


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
        for each epoch, with each epoch indicated by the Index column.
        The analyzed features consist of the mean and minimum
        ECG rate, both adjusted for baseline.

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
                                  epochs_duration=3)
    >>> nk.ecg_eventrelated(epochs)
    >>>
    >>> # Example with real data
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
    >>> nk.ecg_eventrelated(epochs)
    """
    # Sanity checks
    if isinstance(epochs, pd.DataFrame):
        epochs = _df_to_epochs(epochs)  # Convert df to dict

    if not isinstance(epochs, dict):
        raise ValueError("NeuroKit error: ecg_eventrelated(): Please specify an input"
                         "that is of the correct form i.e., either a dictionary"
                         "or dataframe as returned by `epochs_create()`.")

    # Extract features and build dataframe
    ecg_df = {}  # Initialize an empty dict
    for epoch_index in epochs:
        ecg_df[epoch_index] = {}  # Initialize an empty dict for the current epoch
        epoch = epochs[epoch_index]

        # Rate
        ecg_df[epoch_index] = _ecg_eventrelated_rate(epoch)

    ecg_df = pd.DataFrame.from_dict(ecg_df, orient="index")  # Convert to a dataframe

    return ecg_df


# =============================================================================
# Internals
# =============================================================================

def _ecg_eventrelated_rate(epoch):

    output = {}

    # Sanitize input
    colnames = epoch.columns.values
    if len([i for i in colnames if "ECG_Rate" in i]) == 0:
        print("NeuroKit warning: ecg_eventrelated(): input does not"
              "have an `ECG_Rate` column. Will skip all rate-related features.")
        return output

    # Get baseline
    if np.min(epoch.index) <= 0:
        baseline = epoch["ECG_Rate"][epoch.index <= 0].values
        signal = epoch["ECG_Rate"][epoch.index > 0].values
    else:
        baseline = epoch["ECG_Rate"][np.min(epoch.index):np.min(epoch.index)].values
        signal = epoch["ECG_Rate"][epoch.index > np.min(epoch.index)].values


    # Min / Mean
    output["ECG_Rate_Max"] = np.max(signal) - np.mean(baseline)
    output["ECG_Rate_Min"] = np.min(signal) - np.mean(baseline)
    output["ECG_Rate_Mean"] = np.mean(signal) - np.mean(baseline)

    # Modelling
    # TODO
#    nk.fit_polynomial(signal, order=2)
    return output

