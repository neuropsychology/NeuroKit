# -*- coding: utf-8 -*-
import pandas as pd

from ..ecg import ecg_analyze
from ..rsp import rsp_analyze
from ..eda import eda_analyze
from ..emg import emg_analyze


def bio_analyze(data, sampling_rate=1000, method="auto"):
    """Automated analysis of bio signals.

    Wrapper for other bio analyze functions of
    electrocardiography signals (ECG), respiration signals (RSP),
    electrodermal activity (EDA) and electromyography signals (EMG).

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing all the processed signals, typically
        produced by `bio_process()`, `ecg_process()`, `rsp_process()`,
        `eda_process()`, or `emg_process()`.
    sampling_rate : int
        The sampling frequency of the signals (in Hz, i.e., samples/second).
        Defaults to 1000.
    method : str
        Can be one of 'event-related' for event-related analysis on epochs,
        or 'interval-related' for analysis on longer periods of data. Defaults
        to 'auto' where the right method will be chosen based on the
        mean duration of the data ('event-related' for duration under 10s).

    Returns
    ----------
    DataFrame
        DataFrame of the analyzed bio features. See docstrings of `ecg_analyze()`,
        `rsp_analyze()`, `eda_analyze()` and `emg_analyze()` for more details.

    See Also
    ----------
    ecg_analyze, rsp_analyze, eda_analyze, emg_analyze

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> Example 1: Event-related analysis
    >>> # Download data
    >>> data = nk.data("bio_eventrelated_100hz")
    >>>
    >>> # Process the data
    >>> df, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"], eda=data["EDA"], keep=data["Photosensor"], sampling_rate=100)
    >>>
    >>> # Build epochs
    >>> events = nk.events_find(data["Photosensor"],
                                threshold_keep='below',
                                event_conditions=["Negative",
                                                  "Neutral",
                                                  "Neutral",
                                                  "Negative"])
    >>> epochs = nk.epochs_create(df, events,
                                  sampling_rate=100,
                                  epochs_start=-0.1, epochs_end=1.9)
    >>> # Analyze
    >>> nk.bio_analyze(epochs)
    >>>
    >>> Example 2: Interval-related analysis
    >>> # Download data
    >>> data = nk.data("bio_resting_5min_100hz")
    >>>
    >>> # Process the data
    >>> df, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"], sampling_rate=100)
    >>>
    >>> # Analyze
    >>> nk.bio_analyze(df)
    """
    features = pd.DataFrame()
    method = method.lower()

    # Sanitize input
    if isinstance(data, pd.DataFrame):
        ecg_cols = [col for col in data.columns if 'ECG' in col]
        rsp_cols = [col for col in data.columns if 'RSP' in col]
        eda_cols = [col for col in data.columns if 'EDA' in col]
        emg_cols = [col for col in data.columns if 'EMG' in col]
    elif isinstance(data, dict):
        for i in data:
            ecg_cols = [col for col in data[i].columns if 'ECG' in col]
            rsp_cols = [col for col in data[i].columns if 'RSP' in col]
            eda_cols = [col for col in data[i].columns if 'EDA' in col]
            emg_cols = [col for col in data[i].columns if 'EMG' in col]
    else:
        raise ValueError("NeuroKit error: bio_analyze(): Wrong input, "
                         "Please make sure you enter a DataFrame or "
                         "a dictionary. ")

    # ECG
    if len(ecg_cols) != 0:
        ecg_analyzed = ecg_analyze(data, sampling_rate=sampling_rate,
                                   method=method)
        features = pd.concat([features, ecg_analyzed], axis=1)

    # RSP
    if len(rsp_cols) != 0:
        rsp_analyzed = rsp_analyze(data, sampling_rate=sampling_rate,
                                   method=method)
        features = pd.concat([features, rsp_analyzed], axis=1)

    # EDA
    if len(eda_cols) != 0:
        eda_analyzed = eda_analyze(data, sampling_rate=sampling_rate,
                                   method=method)
        features = pd.concat([features, eda_analyzed], axis=1)

    # ECG
    if len(emg_cols) != 0:
        emg_analyzed = emg_analyze(data, sampling_rate=sampling_rate,
                                   method=method)
        features = pd.concat([features, emg_analyzed], axis=1)

    return features
