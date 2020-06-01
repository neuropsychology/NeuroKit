# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..ecg.ecg_analyze import ecg_analyze
from ..ecg.ecg_rsa import ecg_rsa
from ..eda.eda_analyze import eda_analyze
from ..emg.emg_analyze import emg_analyze
from ..rsp.rsp_analyze import rsp_analyze


def bio_analyze(data, sampling_rate=1000, method="auto"):
    """
    Automated analysis of bio signals.

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
        Also returns Respiratory Sinus Arrhythmia features produced by
        `ecg_rsa()` if interval-related analysis is carried out.

    See Also
    ----------
    ecg_analyze, rsp_analyze, eda_analyze, emg_analyze

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Example 1: Event-related analysis
    >>> # Download data
    >>> data = nk.data("bio_eventrelated_100hz")
    >>>
    >>> # Process the data
    >>> df, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"], eda=data["EDA"], keep=data["Photosensor"], sampling_rate=100)
    >>>
    >>> # Build epochs
    >>> events = nk.events_find(data["Photosensor"], threshold_keep='below', event_conditions=["Negative", "Neutral", "Neutral", "Negative"])
    >>> epochs = nk.epochs_create(df, events, sampling_rate=100, epochs_start=-0.1, epochs_end=1.9)
    >>>
    >>> # Analyze
    >>> nk.bio_analyze(epochs) #doctest: +SKIP
    >>>
    >>> # Example 2: Interval-related analysis
    >>> # Download data
    >>> data = nk.data("bio_resting_5min_100hz")
    >>>
    >>> # Process the data
    >>> df, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"], sampling_rate=100)
    >>>
    >>> # Analyze
    >>> nk.bio_analyze(df) #doctest: +SKIP

    """
    features = pd.DataFrame()
    method = method.lower()

    # Sanitize input
    if isinstance(data, pd.DataFrame):
        ecg_cols = [col for col in data.columns if "ECG" in col]
        rsp_cols = [col for col in data.columns if "RSP" in col]
        eda_cols = [col for col in data.columns if "EDA" in col]
        emg_cols = [col for col in data.columns if "EMG" in col]
        ecg_rate_col = [col for col in data.columns if "ECG_Rate" in col]
        rsp_phase_col = [col for col in data.columns if "RSP_Phase" in col]
    elif isinstance(data, dict):
        for i in data:
            ecg_cols = [col for col in data[i].columns if "ECG" in col]
            rsp_cols = [col for col in data[i].columns if "RSP" in col]
            eda_cols = [col for col in data[i].columns if "EDA" in col]
            emg_cols = [col for col in data[i].columns if "EMG" in col]
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

    # RSA
    if len(ecg_rate_col + rsp_phase_col) >= 3:

        # Event-related
        if method in ["event-related", "event", "epoch"]:
            rsa = _bio_analyze_rsa_event(data, sampling_rate=sampling_rate)

        # Interval-related
        elif method in ["interval-related", "interval", "resting-state"]:
            rsa = _bio_analyze_rsa_interval(data, sampling_rate=sampling_rate)

        # Auto
        elif method in ["auto"]:
            if isinstance(data, dict):
                for i in data:
                    duration = len(data[i]) / sampling_rate
                if duration >= 10:
                    rsa = _bio_analyze_rsa_interval(data, sampling_rate=sampling_rate)
                else:
                    rsa = _bio_analyze_rsa_event(data, sampling_rate=sampling_rate)

            if isinstance(data, pd.DataFrame):
                if "Label" in data.columns:
                    epoch_len = data["Label"].value_counts()[0]
                    duration = epoch_len / sampling_rate
                else:
                    duration = len(data) / sampling_rate
                if duration >= 10:
                    rsa = _bio_analyze_rsa_interval(data, sampling_rate=sampling_rate)
                else:
                    rsa = _bio_analyze_rsa_event(data, sampling_rate=sampling_rate)

        features = pd.concat([features, rsa], axis=1, sort=False)

    # Remove duplicate columns of Label and Condition
    if "Label" in features.columns.values:
        features = features.loc[:, ~features.columns.duplicated()]

    return features


# =============================================================================
# Internals
# =============================================================================


def _bio_analyze_rsa_interval(data, sampling_rate=1000):
    # RSA features for interval-related analysis

    if isinstance(data, pd.DataFrame):
        rsa = ecg_rsa(data, sampling_rate=sampling_rate, continuous=False)
        rsa = pd.DataFrame.from_dict(rsa, orient="index").T

    if isinstance(data, dict):
        rsa = {}
        for index in data:
            rsa[index] = {}  # Initialize empty container
            data[index] = data[index].set_index("Index").drop(["Label"], axis=1)
            rsa[index] = ecg_rsa(data[index], sampling_rate=sampling_rate)
        rsa = pd.DataFrame.from_dict(rsa, orient="index")

    return rsa


def _bio_analyze_rsa_event(data, sampling_rate=1000, rsa={}):
    # RSA features for event-related analysis

    if isinstance(data, dict):
        for i in data:
            rsa[i] = {}
            rsa[i] = _bio_analyze_rsa_epoch(data[i], rsa[i])
        rsa = pd.DataFrame.from_dict(rsa, orient="index")

    if isinstance(data, pd.DataFrame):
        rsa = data.groupby("Label")["RSA_P2T"].mean()
        # TODO Needs further fixing

    return rsa


def _bio_analyze_rsa_epoch(epoch, output={}):
    # RSA features for event-related analysis: epoching

    if np.min(epoch.index.values) <= 0:
        baseline = epoch["RSA_P2T"][epoch.index <= 0].values
        signal = epoch["RSA_P2T"][epoch.index > 0].values
        output["RSA_P2T"] = np.mean(signal) - np.mean(baseline)
    else:
        signal = epoch["RSA_P2T"].values
        output["RSA_P2T"] = np.mean(signal)

    return output
