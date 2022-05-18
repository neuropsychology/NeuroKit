# -*- coding: utf-8 -*-
import numpy as np


def signal_flatline(signal, threshold=0.01):
    """**Return the Flatline Percentage of the Signal**

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    threshold : float, optional
        Flatline threshold relative to the biggest change in the signal.
        This is the percentage of the maximum value of absolute consecutive
        differences.

    Returns
    -------
    float
        Percentage of signal where the absolute value of the derivative is lower then the threshold.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=5)
      nk.signal_flatline(signal)


    """
    diff = np.diff(signal)
    threshold = threshold * np.max(np.abs(diff))

    flatline = np.where(np.abs(diff) < threshold)[0]

    return len(flatline) / len(signal)
