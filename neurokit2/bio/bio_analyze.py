# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..ecg import ecg_analyze
from ..hrv import hrv_rsa
from ..eda import eda_analyze
from ..emg import emg_analyze
from ..eog import eog_analyze
from ..rsp import rsp_analyze


def bio_analyze(data, sampling_rate=1000, method="auto"):
    """Automated analysis of bio signals.

    Wrapper for other bio analyze functions of
    electrocardiography signals (ECG), respiration signals (RSP), electrodermal activity (EDA),
    electromyography signals (EMG) and electrooculography signals (EOG).

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing all the processed signals, typically
        produced by `bio_process()`, `ecg_process()`, `rsp_process()`,
        `eda_process()`, `emg_process()` or `eog_process()`.
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
        `rsp_analyze()`, `eda_analyze()`, `emg_analyze()` and `eog_analyze()` for more details.
        Also returns Respiratory Sinus Arrhythmia features produced by
        `hrv_rsa()` if interval-related analysis is carried out.

    See Also
    ----------
    ecg_analyze, rsp_analyze, eda_analyze, emg_analyze, eog_analyze

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Example 1: Event-related analysis
    >>> # Download data
    >>> data = nk.data("bio_eventrelated_100hz")
    >>>
    >>> # Process the data
    >>> df, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"], eda=data["EDA"],
    ...                           keep=data["Photosensor"], sampling_rate=100)
    >>>
    >>> # Build epochs
    >>> events = nk.events_find(data["Photosensor"], threshold_keep='below',
    ...                         event_conditions=["Negative", "Neutral",
    ...                                           "Neutral", "Negative"])
    >>> epochs = nk.epochs_create(df, events, sampling_rate=100, epochs_start=-0.1,
    ...                           epochs_end=1.9)
    >>>
    >>> # Analyze
    >>> nk.bio_analyze(epochs, sampling_rate=100) #doctest: +ELLIPSIS
      Label Condition  Event_Onset  ...         RSA_Gates
    1     1  Negative          ...  ...           ...
    2     2   Neutral          ...  ...           ...
    3     3   Neutral          ...  ...           ...
    4     4  Negative          ...  ...           ...

    [4 rows x 39 columns]
    >>>
    >>> # Example 2: Interval-related analysis
    >>> # Download data
    >>> data = nk.data("bio_resting_5min_100hz")
    >>>
    >>> # Process the data
    >>> df, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"], sampling_rate=100)
    >>>
    >>> # Analyze
    >>> nk.bio_analyze(df, sampling_rate=100) #doctest: +ELLIPSIS
       ECG_Rate_Mean  HRV_RMSSD  ...  RSA_Gates_Mean_log  RSA_Gates_SD
    0            ...        ...  ...            ...               ...

    [1 rows x 84 columns]
    """
    features = pd.DataFrame()
    method = method.lower()

    # Sanitize input
    if isinstance(data, pd.DataFrame):
        ecg_cols = [col for col in data.columns if "ECG" in col]
        rsp_cols = [col for col in data.columns if "RSP" in col]
        eda_cols = [col for col in data.columns if "EDA" in col]
        emg_cols = [col for col in data.columns if "EMG" in col]
        eog_cols = [col for col in data.columns if "EOG" in col]
        ecg_rate_col = [col for col in data.columns if "ECG_Rate" in col]
        rsp_phase_col = [col for col in data.columns if "RSP_Phase" in col]
    elif isinstance(data, dict):
        for i in data:
            ecg_cols = [col for col in data[i].columns if "ECG" in col]
            rsp_cols = [col for col in data[i].columns if "RSP" in col]
            eda_cols = [col for col in data[i].columns if "EDA" in col]
            emg_cols = [col for col in data[i].columns if "EMG" in col]
            eog_cols = [col for col in data[i].columns if "EOG" in col]
            ecg_rate_col = [col for col in data[i].columns if "ECG_Rate" in col]
            rsp_phase_col = [col for col in data[i].columns if "RSP_Phase" in col]
    else:
        raise ValueError(
            "NeuroKit error: bio_analyze(): Wrong input, please make sure you enter a DataFrame or a dictionary. "
        )

    # ECG
    ecg_data = data.copy()
    if len(ecg_cols) != 0:
        ecg_analyzed = ecg_analyze(ecg_data, sampling_rate=sampling_rate, method=method)
        features = pd.concat([features, ecg_analyzed], axis=1, sort=False)

    # RSP
    rsp_data = data.copy()
    if len(rsp_cols) != 0:
        rsp_analyzed = rsp_analyze(rsp_data, sampling_rate=sampling_rate, method=method)
        features = pd.concat([features, rsp_analyzed], axis=1, sort=False)

    # EDA
    if len(eda_cols) != 0:
        eda_analyzed = eda_analyze(data, sampling_rate=sampling_rate, method=method)
        features = pd.concat([features, eda_analyzed], axis=1, sort=False)

    # EMG
    if len(emg_cols) != 0:
        emg_analyzed = emg_analyze(data, sampling_rate=sampling_rate, method=method)
        features = pd.concat([features, emg_analyzed], axis=1, sort=False)

    # EOG
    if len(eog_cols) != 0:
        eog_analyzed = eog_analyze(data, sampling_rate=sampling_rate, method=method)
        features = pd.concat([features, eog_analyzed], axis=1, sort=False)

    # RSA
    if len(ecg_rate_col + rsp_phase_col) >= 3:

        # Event-related
        if method in ["event-related", "event", "epoch"]:
            rsa = _bio_analyze_rsa_event(data)

        # Interval-related
        elif method in ["interval-related", "interval", "resting-state"]:
            rsa = _bio_analyze_rsa_interval(data, sampling_rate=sampling_rate)

        # Auto
        else:
            duration = _bio_analyze_findduration(data, sampling_rate=sampling_rate)
            if duration >= 10:
                rsa = _bio_analyze_rsa_interval(data, sampling_rate=sampling_rate)
            else:
                rsa = _bio_analyze_rsa_event(data)

        features = pd.concat([features, rsa], axis=1, sort=False)

    # Remove duplicate columns of Label and Condition
    if "Label" in features.columns.values:
        features = features.loc[:, ~features.columns.duplicated()]

    return features


# =============================================================================
# Internals
# =============================================================================
def _bio_analyze_findduration(data, sampling_rate=1000):
    # If DataFrame
    if isinstance(data, pd.DataFrame):
        if "Label" in data.columns:
            labels = data["Label"].unique()
            durations = [len(data[data["Label"] == label]) / sampling_rate for label in labels]
        else:
            durations = [len(data) / sampling_rate]

    # If dictionary
    if isinstance(data, dict):
        durations = [len(data[i]) / sampling_rate for i in data]

    return np.nanmean(durations)


def _bio_analyze_rsa_interval(data, sampling_rate=1000):
    # RSA features for interval-related analysis


    if isinstance(data, pd.DataFrame):
        rsa = hrv_rsa(data, sampling_rate=sampling_rate, continuous=False)
        rsa = pd.DataFrame.from_dict(rsa, orient="index").T

    elif isinstance(data, dict):
        for index in data:
            rsa[index] = {}  # Initialize empty container
            data[index] = data[index].set_index("Index").drop(["Label"], axis=1)
            rsa[index] = hrv_rsa(data[index], sampling_rate=sampling_rate, continuous=False)
        rsa = pd.DataFrame.from_dict(rsa, orient="index")

    return rsa


def _bio_analyze_rsa_event(data, rsa={}):
    # RSA features for event-related analysis

    if isinstance(data, dict):
        for i in data:
            rsa[i] = {}
            rsa[i] = _bio_analyze_rsa_epoch(data[i], rsa[i])
        rsa = pd.DataFrame.from_dict(rsa, orient="index")

    elif isinstance(data, pd.DataFrame):
        rsa["RSA_P2T"] = np.nanmean(data.groupby("Label")["RSA_P2T"])
        rsa["RSA_Gates"] = np.nanmean(data.groupby("Label")["RSA_Gates"])
        # TODO Needs further fixing

    return rsa


def _bio_analyze_rsa_epoch(epoch, output={}):
    # RSA features for event-related analysis: epoching

    # To remove baseline
    if np.min(epoch.index.values) <= 0:
        baseline = epoch["RSA_P2T"][epoch.index <= 0].values
        signal = epoch["RSA_P2T"][epoch.index > 0].values
        output["RSA_P2T"] = np.mean(signal) - np.mean(baseline)
        baseline = epoch["RSA_Gates"][epoch.index <= 0].values
        signal = epoch["RSA_Gates"][epoch.index > 0].values
        output["RSA_Gates"] = np.nanmean(signal) - np.nanmean(baseline)
    else:
        signal = epoch["RSA_P2T"].values
        output["RSA_P2T"] = np.mean(signal)
        signal = epoch["RSA_Gates"].values
        output["RSA_Gates"] = np.nanmean(signal)

    return output
