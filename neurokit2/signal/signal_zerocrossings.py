# -*- coding: utf-8 -*-
import numpy as np


def signal_zerocrossings(signal, direction="both"):
    """Locate the indices where the signal crosses zero.

    Note that when the signal crosses zero between two points, the first index is returned.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    direction : str
        Direction in which the signal crosses zero, can be "positive", "negative" or "both" (default).

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
    >>>
    >>> # Only upward or downward zerocrossings
    >>> up = nk.signal_zerocrossings(signal, direction='up')
    >>> down = nk.signal_zerocrossings(signal, direction='down')
    >>> fig = nk.events_plot([up, down], signal)
    >>> fig #doctest: +SKIP

    """
    df = np.diff(np.sign(signal))
    if direction in ["positive", "up"]:
        zerocrossings = np.where(df > 0)[0]
    elif direction in ["negative", "down"]:
        zerocrossings = np.where(df < 0)[0]
    else:
        zerocrossings = np.nonzero(np.abs(df) > 0)[0]

    return zerocrossings
