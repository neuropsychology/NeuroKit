# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pandas as pd

from ..signal import signal_period


def benchmark_ecg_preprocessing(function, ecg, rpeaks=None, sampling_rate=1000):
    """Benchmark ECG preprocessing pipelines.

    Parameters
    ----------
    function : function
        Must be a Python function which first argument is the ECG signal and which has a
        ``sampling_rate`` argument.
    ecg : pd.DataFrame or str
        The path to a folder where you have an `ECGs.csv` file or directly its loaded DataFrame.
        Such file can be obtained by running THIS SCRIPT (TO COMPLETE).
    rpeaks : pd.DataFrame or str
        The path to a folder where you have an `Rpeaks.csv` fils or directly its loaded DataFrame.
        Such file can be obtained by running THIS SCRIPT (TO COMPLETE).
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second). Only used if ``ecgs``
        and ``rpeaks`` are single vectors.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing the results of the benchmarking


    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Define a preprocessing routine
    >>> def function(ecg, sampling_rate):
    >>>     signal, info = nk.ecg_peaks(ecg, method='engzeemod2012', sampling_rate=sampling_rate)
    >>>     return info["ECG_R_Peaks"]
    >>>
    >>> # Synthetic example
    >>> ecg = nk.ecg_simulate(duration=20, sampling_rate=200)
    >>> true_rpeaks = nk.ecg_peaks(ecg, sampling_rate=200)[1]["ECG_R_Peaks"]
    >>>
    >>> nk.benchmark_ecg_preprocessing(function, ecg, true_rpeaks, sampling_rate=200)
    >>>
    >>> # Example using database (commented-out)
    >>> # nk.benchmark_ecg_preprocessing(function, r'path/to/GUDB_database')

    """
    # find data
    if rpeaks is None:
        rpeaks = ecg

    if isinstance(ecg, str):
        ecg = pd.read_csv(ecg + "/ECGs.csv")

    if isinstance(rpeaks, str):
        rpeaks = pd.read_csv(rpeaks + "/Rpeaks.csv")

    if isinstance(ecg, pd.DataFrame):
        results = _benchmark_ecg_preprocessing_databases(function, ecg, rpeaks)
    else:
        results = _benchmark_ecg_preprocessing(function, ecg, rpeaks, sampling_rate=sampling_rate)

    return results


# =============================================================================
# Utils
# =============================================================================
def _benchmark_ecg_preprocessing_databases(function, ecgs, rpeaks):
    """A wrapper over _benchmark_ecg_preprocessing when the input is a database."""
    # Run algorithms
    results = []
    for participant in ecgs["Participant"].unique():
        for database in ecgs[ecgs["Participant"] == participant]["Database"].unique():

            # Extract the right slice of data
            ecg_slice = ecgs[(ecgs["Participant"] == participant) & (ecgs["Database"] == database)]
            rpeaks_slice = rpeaks[(rpeaks["Participant"] == participant) & (rpeaks["Database"] == database)]
            sampling_rate = ecg_slice["Sampling_Rate"].unique()[0]

            # Extract values
            ecg = ecg_slice["ECG"].values
            rpeak = rpeaks_slice["Rpeaks"].values

            # Run benchmark
            result = _benchmark_ecg_preprocessing(function, ecg, rpeak, sampling_rate)

            # Add info
            result["Participant"] = participant
            result["Database"] = database

            results.append(result)

    return pd.concat(results)


def _benchmark_ecg_preprocessing(function, ecg, rpeak, sampling_rate=1000):
    # Apply function
    t0 = datetime.datetime.now()
    try:
        found_rpeaks = function(ecg, sampling_rate=sampling_rate)
        duration = (datetime.datetime.now() - t0).total_seconds()
    # In case of failure
    except Exception as error:  # pylint: disable=broad-except
        return pd.DataFrame(
            {
                "Sampling_Rate": [sampling_rate],
                "Duration": [np.nan],
                "Score": [np.nan],
                "Recording_Length": [len(ecg) / sampling_rate / 60],
                "Error": str(error),
            }
        )

    # Compare R peaks
    score, error = benchmark_ecg_compareRpeaks(rpeak, found_rpeaks, sampling_rate=sampling_rate)

    return pd.DataFrame(
        {
            "Sampling_Rate": [sampling_rate],
            "Duration": [duration],
            "Score": [score],
            "Recording_Length": [len(ecg) / sampling_rate / 60],
            "Error": error,
        }
    )


# =============================================================================
# Comparison methods
# =============================================================================
def benchmark_ecg_compareRpeaks(true_rpeaks, found_rpeaks, sampling_rate=250):
    # Failure to find sufficient R-peaks
    if len(found_rpeaks) <= 3:
        return np.nan, "R-peaks detected <= 3"

    length = np.max(np.concatenate([true_rpeaks, found_rpeaks]))

    true_interpolated = signal_period(
        true_rpeaks, sampling_rate=sampling_rate, desired_length=length, interpolation_method="linear"
    )
    found_interpolated = signal_period(
        found_rpeaks, sampling_rate=sampling_rate, desired_length=length, interpolation_method="linear"
    )

    return np.mean(np.abs(found_interpolated - true_interpolated)), "None"
