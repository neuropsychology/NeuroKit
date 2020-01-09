# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal import signal_interpolate


def signal_rate(peaks, sampling_rate=1000, desired_length=None):
    """Calculate signal rate from a series of peaks.

    Parameters
    ----------
    peaks : list, array, DataFrame, Series or dict
        The samples at which thepeaks occur. If an array is
        passed, it is assumed that these containers were obtained with
        `signal_findpeaks()`. If a DataFrame is passed, it is assumed it is of the same length as
        the input signal in which occurrences of R-peaks are marked as "1", with such containers
        obtained with e.g., ecg_findpeaks() or rsp_findpeaks().
    sampling_rate : int
        The sampling frequency of the signal that contains the R-peaks (in Hz,
        i.e., samples/second). Defaults to 1000.
    desired_length : int
        By default, the returned signal rate has the same number of elements as
        the raw signal. If set to an integer, the returned signal rate will be
        interpolated between R-peaks over `desired_length` samples. Has no
        effect if a DataFrame is passed in as the `signal` argument. Defaults to
        None.

    Returns
    -------
    array
        A vector containing the rate.

    See Also
    --------
    signal_findpeaks, signal_plot, rsp_rate, ecg_rate

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=10, sampling_rate=1000, frequency=1)
    >>> peaks, info = nk.signal_findpeaks(signal)
    >>>
    >>> rate = nk.signal_rate(peaks)
    >>> nk.signal_plot(rate)
    """
    # Format input.
    if desired_length is None:
        if isinstance(peaks, np.ndarray):
            desired_length = max(peaks)
        elif isinstance(peaks, pd.DataFrame):
            desired_length = len(peaks)
            # Attempt to retrieve column
            col = [col for col in peaks.columns if 'Peaks' in col]
            if len(col) == 0:
                TypeError("NeuroKit error: signal_rate(): wrong type of input ",
                          "provided. Please provide indices of peaks.")
            peaks_signal = peaks[col[0]].values
            peaks = np.where(peaks_signal == 1)[0]
    elif desired_length < len(peaks):
        raise ValueError("NeuroKit error: signal_rate(): 'desired_length' cannot",
                         " be lower than the length of the signal. Please input a greater 'desired_length'.")

    # Determine length of final signal to return.
    if desired_length is None:
        desired_length = len(peaks)

    # Sanity checks.
    if len(peaks) <= 3:
        print("NeuroKit warning: signal_rate(): too few peaks detected to "
              "compute the rate. Returning empty vector.")
        return np.full(desired_length, np.nan)

    # Calculate period in msec, based on peak to peak difference and make sure
    # that rate has the same number of elements as peaks (important for
    # interpolation later) by prepending the mean of all periods.
    period = np.ediff1d(peaks, to_begin=0) / sampling_rate
    period[0] = np.mean(period[1::])
    rate = 60 / period

    # Interpolate all statistics to desired length.
    if desired_length != len(peaks):
        rate = signal_interpolate(rate, x_axis=peaks, desired_length=desired_length)

    return rate
