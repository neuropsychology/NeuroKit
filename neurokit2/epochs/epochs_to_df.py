# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def epochs_to_df(epochs):
    """Convert epochs to a DataFrame.

    Parameters
    ----------
    epochs : dict
        A dict containing one DataFrame per event/trial. Usually obtained via `epochs_create()`.


    Returns
    ----------
    DataFrame
        A DataFrame containing all epochs identifiable by the 'Label' column, which time axis
        is stored in the 'Time' column.


    See Also
    ----------
    events_find, events_plot, epochs_create, epochs_plot

    Examples
    ----------
    >>> import neurokit2 as nk
    >>> import pandas as pd
    >>>
    >>> # Get data
    >>> data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/dev/data/bio_eventrelated_100hz.csv")
    >>>
    >>> # Find events
    >>> events = nk.events_find(data["Photosensor"],
    ...                         threshold_keep='below',
    ...                         event_conditions=["Negative", "Neutral", "Neutral", "Negative"])
    >>> fig = nk.events_plot(events, data)
    >>> fig #doctest: +SKIP
    >>>
    >>> # Create epochs
    >>> epochs = nk.epochs_create(data, events, sampling_rate=200, epochs_end=3)
    >>> data = nk.epochs_to_df(epochs)

    """
    data = pd.concat(epochs)
    data["Time"] = data.index.get_level_values(1).values
    data = data.reset_index(drop=True)

    return data


def _df_to_epochs(data):
    # Convert dataframe of epochs created by `epochs_to_df` back into a dictionary.
    labels = data.Label.unique()
    epochs_dict = {i: pd.DataFrame for i in labels}
    for key in epochs_dict:
        epochs_dict[key] = data[:][data.Label == key]
        epochs_dict[key].index = np.array(epochs_dict[key]["Time"])
        epochs_dict[key] = epochs_dict[key].drop(["Time"], axis=1)

    return epochs_dict
