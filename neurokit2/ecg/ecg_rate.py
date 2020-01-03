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
    >>>
    """
    if isinstance(peaks, dict):
        peaks = peaks["ECG_Peaks"]
    elif isinstance(peaks, pd.DataFrame):
        desired_length = len(peaks["ECG_Peaks"])
        peaks = np.where(peaks["ECG_Peaks"] == 1)[0]

    # Determine length of final signal to return.
    if desired_length is None:
        desired_length = len(peaks)

    # Sanity checks.
    if len(peaks) <= 3:
        print("NeuroKit warning: ecg_rate(): too few peaks detected to "
              "compute the rate.")
        return

    # Calculate period in msec, based on peak to peak difference and make sure
    # that rate has the same number of elements as peaks (important for
    # interpolation later) by prepending the mean of all periods.
    period = np.ediff1d(peaks, to_begin=0) / sampling_rate
    period[0] = np.mean(period)
    # Get rate.
    rate = 60 / period

    # Interpolate all statistics to desired lenght.
    rate = signal_interpolate(rate, x_axis=peaks,
                              desired_length=desired_length)

    # Prepare output
    signals = pd.DataFrame(rate, columns=["ECG_Rate"])
    return(signals)
