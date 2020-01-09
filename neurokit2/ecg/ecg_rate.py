# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal import signal_interpolate


def ecg_rate(peaks, sampling_rate=1000, desired_length=None):
    """Calculate heart rate from R-peaks.

    Parameters
    ----------
    peaks : list, array, DataFrame, Series or dict
        The samples at which the R-peaks occur. If a dict or a DataFrame is
        passed, it is assumed that these containers were obtained with
        `ecg_findpeaks()`.
    sampling_rate : int
        The sampling frequency of the signal that contains the R-peaks (in Hz,
        i.e., samples/second). Defaults to 1000.
    desired_length : int
        By default, the returned heart rate has the same number of elements as
        peaks. If set to an integer, the returned heart rate will be
        interpolated between R-peaks over `desired_length` samples. Has no
        effect if a DataFrame is passed in as the `peaks` argument. Defaults to
        None.

    Returns
    -------
    signals : DataFrame
        A DataFrame containing heart rate accessible with the key "ECG_Rate".

    See Also
    --------
    ecg_clean, ecg_findpeaks, ecg_process, ecg_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> ecg = nk.ecg_simulate(duration=15, heart_rate=80)
    >>> signals, info = nk.ecg_findpeaks(ecg)
    >>>
    >>> data = nk.ecg_rate(signals)
    >>> data["ECG_Signal"] = ecg  # Add the signal back
    >>> data.plot(subplots=True)
    """
    # Get rate values
    rate = signal_rate(peaks, sampling_rate, desired_length=desired_length)


    # Prepare output
    signals = pd.DataFrame(rate, columns=["ECG_Rate"])
    return signals
