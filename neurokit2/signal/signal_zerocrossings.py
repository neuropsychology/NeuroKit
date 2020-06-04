# -*- coding: utf-8 -*-
import numpy as np


def signal_zerocrossings(signal):
    """
    Locate the indices where the signal crosses zero.

    Note that when the signal crosses zero between two points, the first index is returned.

    Parameters
    ----------
    signal : list, array or Series
        The signal (i.e., a time series) in the form of a vector of values.

    Returns
    -------
    array
        Vector containing the indices of zero crossings.

    Examples
    --------
    >>> import numpy as np
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=15, num=1000))
    >>> zeros = nk.signal_zerocrossings(signal)
    >>> fig = nk.events_plot(zeros, signal)
    >>> fig #doctest: +SKIP

    """
    df = np.diff(np.sign(signal))
    zeros = np.nonzero(np.abs(df) > 0)[0]

    return zeros
