# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .signal_resample import signal_resample


def signal_merge(signal1, signal2, time1=[0, 10], time2=[0, 10]):
    """
    Arbitrary addition of two signals with different time ranges.

    Parameters
    ----------
    signal1, signal2 : list, array or Series
        The signal (i.e., a time series)s in the form of a vector of values.
    time1, time2 : list
        Lists containing two numeric values corresponding to the beginning and end of 'signal1' and 'signal2', respectively.

    Returns
    -------
    array
        Vector containing the sum of the two signals.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal1 = np.cos(np.linspace(start=0, stop=10, num=100))
    >>> signal2 = np.cos(np.linspace(start=0, stop=20, num=100))
    >>>
    >>> signal = nk.signal_merge(signal1, signal2, time1=[0, 10], time2=[-5, 5])
    >>> nk.signal_plot(signal)

    """

    # Resample signals if different
    sampling_rate1 = len(signal1) / np.diff(time1)[0]
    sampling_rate2 = len(signal2) / np.diff(time2)[0]
    if sampling_rate1 > sampling_rate2:
        signal2 = signal_resample(signal2, sampling_rate=sampling_rate2, desired_sampling_rate=sampling_rate1)
    elif sampling_rate2 > sampling_rate1:
        signal1 = signal_resample(signal1, sampling_rate=sampling_rate1, desired_sampling_rate=sampling_rate2)
    sampling_rate = np.max([sampling_rate1, sampling_rate2])

    # Fill beginning
    if time1[0] < time2[0]:
        beginning = np.full(int(np.round(sampling_rate * (time2[0] - time1[0]))), signal2[0])
        signal2 = np.concatenate((beginning, signal2))
    elif time2[0] < time1[0]:
        beginning = np.full(int(np.round(sampling_rate * (time1[0] - time2[0]))), signal1[0])
        signal1 = np.concatenate((beginning, signal1))

    # Fill end
    if time1[1] > time2[1]:
        end = np.full(int(np.round(sampling_rate * (time1[1] - time2[1]))), signal2[-1])
        signal2 = np.concatenate((signal2, end))
    elif time2[1] > time1[1]:
        end = np.full(int(np.round(sampling_rate * (time2[1] - time1[1]))), signal1[-1])
        signal1 = np.concatenate((signal1, end))

    # Sanitize length of arrays
    if len(signal1) > len(signal2):
        signal1 = signal1[0 : len(signal2)]
    if len(signal2) > len(signal1):
        signal2 = signal2[0 : len(signal1)]

    merged = signal1 + signal2
    return merged
