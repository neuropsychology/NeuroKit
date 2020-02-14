# -*- coding: utf-8 -*-
import pandas as pd

from ..epochs import _df_to_epochs


def signal_erp(epochs):
    """Performs event-related analysis on epochs.

    Parameters
    ----------
    epochs : dict, DataFrame
        A dict containing one DataFrame per event/trial,
        usually obtained via `epochs_create()`, or a DataFrame
        containing all epochs, usually obtained via `epochs_to_df()`.

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
    >>> import neurokit2 as nk
    >>>
    >>> # Simulate signal
    >>> signal = nk.signal_simulate(duration=10, frequency=1)
    >>> events = nk.signal_findpeaks(signal)
    >>>
    >>> # Create epochs
    >>> epochs = nk.epochs_create(signal, events=events["Peaks"], epochs_duration=5, epochs_start=-0.1)
    >>> nk.signal_erp(epochs)
    """
    # Sanity checks
    if isinstance(epochs, pd.DataFrame):
        epochs = _df_to_epochs(epochs)  # Convert df to dict

    if not isinstance(epochs, dict):
        raise ValueError("NeuroKit error: signal_erp(): Please specify an input"
                         "that is of the correct form i.e., either a dictionary"
                         "or dataframe.")

    # Extract features and build dataframe
    df = {}  # Initialize an empty dict
    for epoch_index in epochs:
        df[epoch_index] = {}  # Initialize an empty dict for the current epoch
        epoch = epochs[epoch_index]

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
