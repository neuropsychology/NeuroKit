# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pandas as pd

from ..signal import signal_period


def benchmark_ecg_preprocessing(function, ecgs, rpeaks):
    """Benchmark ECG preprocessing pipelines.

    Parameters
    ----------
    function : function
        Must be a Python function which first argument is the ECG signal and which has a
        ``sampling_rate`` argument.
    ecgs : pd.DataFrame or str
        The path to a folder where you have an `ECGs.csv` file or directly its loaded DataFrame.
        Such file can be obtained by running THIS SCRIPT (TO COMPLETE).
    rpeaks : pd.DataFrame or str
        The path to a folder where you have an `Rpeaks.csv` fils or directly its loaded DataFrame.
        Such file can be obtained by running THIS SCRIPT (TO COMPLETE).

    Returns
    --------
    pd.DataFrame
        A DataFrame containing the results of the benchmarking


    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> def function(ecg, sampling_rate):
    >>>     cleaned = nk.ecg_clean(ecg, sampling_rate=sampling_rate)
    >>>     signal, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
    >>>     return info["ECG_R_Peaks"]
    >>>
    >>> nk.benchmark_ecg(function, path_to_database='path_to_GUDB_database')

    """
    if isinstance(ecgs, str):
        ecgs = pd.read_csv(ecgs + "ECGs.csv")

    if isinstance(rpeaks, str):
        rpeaks = pd.read_csv(rpeaks + "Rpeaks.csv")

    results = []
    for participant in ecgs["Participant"].unique():
        for database in ecgs[ecgs["Participant"] == participant]["Database"].unique():

            # Extract the right slice of data
            ecg = ecgs[(ecgs["Participant"] == participant) & (ecgs["Database"] == database)]
            true_rpeaks = rpeaks[(rpeaks["Participant"] == participant) & (rpeaks["Database"] == database)][
                "Rpeaks"
            ].values

            sampling_rate = ecg["Sampling_Rate"].unique()[0]

            # Apply function
            t0 = datetime.datetime.now()
            found_rpeaks = function(ecg["ECG"].values, sampling_rate=sampling_rate)
            duration = (datetime.datetime.now() - t0).total_seconds()

            # Compare R peaks
            score = benchmark_ecg_compareRpeaks(true_rpeaks, found_rpeaks, sampling_rate=sampling_rate)

            results.append(
                pd.DataFrame(
                    {
                        "Participant": [participant],
                        "Database": [database],
                        "Sampling_Rate": [sampling_rate],
                        "Duration": [duration],
                        "Score": [score],
                        "Recording_Length": [len(ecg) / sampling_rate / 60],
                    }
                )
            )

    return pd.concat(results)


# =============================================================================
# Utils
# =============================================================================
def benchmark_ecg_compareRpeaks(true_rpeaks, found_rpeaks, sampling_rate=250):
    length = np.max(np.concatenate([true_rpeaks, found_rpeaks]))

    true_interpolated = signal_period(
        true_rpeaks, sampling_rate=sampling_rate, desired_length=length, interpolation_order="linear"
    )
    found_interpolated = signal_period(
        found_rpeaks, sampling_rate=sampling_rate, desired_length=length, interpolation_order="linear"
    )

    return np.mean(found_interpolated - true_interpolated)
