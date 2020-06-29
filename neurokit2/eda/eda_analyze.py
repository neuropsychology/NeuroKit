# -*- coding: utf-8 -*-
import pandas as pd

from .eda_eventrelated import eda_eventrelated
from .eda_intervalrelated import eda_intervalrelated


def eda_analyze(data, sampling_rate=1000, method="auto"):
    """Performs EDA analysis on either epochs (event-related analysis) or on longer periods of data such as resting-
    state data.

    Parameters
    ----------
    data : Union[dict, pd.DataFrame]
        A dictionary of epochs, containing one DataFrame per epoch, usually obtained via `epochs_create()`,
        or a DataFrame containing all epochs, usually obtained via `epochs_to_df()`.
        Can also take a DataFrame of processed signals from a longer period of data, typically generated
        by `eda_process()` or `bio_process()`. Can also take a dict containing sets of separate periods
        of data.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
        Defaults to 1000Hz.
    method : str
        Can be one of 'event-related' for event-related analysis on epochs, or 'interval-related'
        for analysis on longer periods of data. Defaults to 'auto' where the right method will be
        chosen based on the mean duration of the data ('event-related' for duration under 10s).

    Returns
    -------
    DataFrame
        A dataframe containing the analyzed EDA features. If event-related analysis is conducted,
        each epoch is indicated by the `Label` column. See `eda_eventrelated()` and
        `eda_intervalrelated()` docstrings for details.

    See Also
    --------
    bio_process, eda_process, epochs_create, eda_eventrelated, eda_intervalrelated

    Examples
    ----------
    >>> import neurokit2 as nk

    >>> # Example 1: Download the data for event-related analysis
    >>> data = nk.data("bio_eventrelated_100hz")
    >>>
    >>> # Process the data for event-related analysis
    >>> df, info = nk.bio_process(eda=data["EDA"], sampling_rate=100)
    >>> events = nk.events_find(data["Photosensor"], threshold_keep='below',
    ...                         event_conditions=["Negative", "Neutral", "Neutral", "Negative"])
    >>> epochs = nk.epochs_create(df, events, sampling_rate=100, epochs_start=-0.1, epochs_end=1.9)
    >>>
    >>> # Analyze
    >>> nk.eda_analyze(epochs, sampling_rate=100) #doctest: +SKIP
    >>>
    >>> # Example 2: Download the resting-state data
    >>> data = nk.data("bio_resting_8min_100hz")
    >>>
    >>> # Process the data
    >>> df, info = nk.eda_process(data["EDA"], sampling_rate=100)
    >>>
    >>> # Analyze
    >>> nk.eda_analyze(df, sampling_rate=100) #doctest: +SKIP

    """
    method = method.lower()

    # Event-related analysis
    if method in ["event-related", "event", "epoch"]:
        # Sanity checks
        if isinstance(data, dict):
            for i in data:
                colnames = data[i].columns.values
        elif isinstance(data, pd.DataFrame):
            colnames = data.columns.values

        if len([i for i in colnames if "Label" in i]) == 0:
            raise ValueError(
                "NeuroKit error: eda_analyze(): Wrong input or method, we couldn't extract epochs features."
            )
        else:
            features = eda_eventrelated(data)

    # Interval-related analysis
    elif method in ["interval-related", "interval", "resting-state"]:
        features = eda_intervalrelated(data)

    # Auto
    elif method in ["auto"]:

        if isinstance(data, dict):
            for i in data:
                duration = len(data[i]) / sampling_rate
            if duration >= 10:
                features = eda_intervalrelated(data)
            else:
                features = eda_eventrelated(data)

        if isinstance(data, pd.DataFrame):
            if "Label" in data.columns:
                epoch_len = data["Label"].value_counts()[0]
                duration = epoch_len / sampling_rate
            else:
                duration = len(data) / sampling_rate
            if duration >= 10:
                features = eda_intervalrelated(data)
            else:
                features = eda_eventrelated(data)

    return features
