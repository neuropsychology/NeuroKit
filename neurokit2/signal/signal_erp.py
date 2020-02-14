# -*- coding: utf-8 -*-
import pandas as pd


def signal_erp(epochs):
    """Performs event-related analysis on epochs.

    Parameters
    ----------
    epochs : dict
        A dict containing one DataFrame per event/trial.
        Usually obtained via `epochs_create()`.

    Returns
    -------
    DataFrame
        A dataframe containing the analyzed features
        (i.e., the maximum, minimum, time of maximum, time of minimum)
        for each epoch, with each epoch indicated by the Index column.

    See Also
    --------
    events_find, epochs_create

    Examples
    --------
    >>> # Simulate signal
    >>> signal = nk.signal_simulate(duration=10, frequency=1)
    >>> events = nk.signal_findpeaks(signal)
    >>>
    >>> # Create epochs
    >>> epochs = nk.epochs_create(signal, events=events["Peaks"], epochs_duration=3, epochs_start=-0.1)
    >>> signal_erp(epochs)
    """
    # Extract features and build dataframe
    df = {}  # Initialize an empty dict
    for epoch_index in epochs:
        df[epoch_index] = {}  # Initialize an empty dict for the current epoch
        epoch = epochs[epoch_index]

        # ECG
        minimum = epoch["Signal"].loc[0:len(epoch)].min()
        maximum = epoch["Signal"].loc[0:len(epoch)].max()

        t_max = epoch["Signal"].idxmax()
        t_min = epoch["Signal"].idxmin()

        df[epoch_index]["Maximum"] = maximum
        df[epoch_index]["Minimum"] = minimum
        df[epoch_index]["Time of Maximum"] = t_max
        df[epoch_index]["Time of Minimum"] = t_min

    df = pd.DataFrame.from_dict(df, orient="index")  # Convert to a dataframe

    return df
