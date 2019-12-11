# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def signal_inversions(signal):
    """Find inversions (peaks and troughs) in a signal.

    Locate peaks and troughs (local maxima and minima) in a signal using the derivative (the gradient).

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.

    Returns
    ----------
    array
        array containing the inversions indices (as relative to the given signal).
        For instance, the value 3 means that the third sample of the signal is a peak or a troughs.

    Examples
    ---------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    >>> peaks = nk.signal_findpeaks(signal)
    >>> nk.plot_events_in_signal(signal, peaks)
    """
    derivative = np.gradient(signal, 2)
    peaks = np.where(np.diff(np.sign(derivative)))[0]
    return(peaks)