import numpy as np


def signal_autocor(signal, lag=None, normalize=True):
    """Auto-correlation of a 1-dimensional sequences.

    Parameters
    -----------
    signal : Union[list, np.array, pd.Series]
        Vector of values.
    normalize : bool
        Normalize the autocorrelation output.
    lag : int
        Time lag.
        If specified, one value of autocorrelation between signal with its lag self will be returned.

    Returns
    -------
    r
        The cross-correlation of the signal with itself at different time lags. Minimum time lag is 0,
        maximum time lag is the length of the signal. Or a correlation value at a specific lag if lag
        is not None.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> x = [1, 2, 3, 4, 5]
    >>> autocor = nk.signal_autocor(x)
    >>> autocor #doctest: +SKIP

    """
    r = np.correlate(signal, signal, mode="full")

    r = r[r.size // 2 :]  # min time lag is 0

    if normalize is True:
        r = r / r[0]

    if lag is not None:
        if lag > len(signal):
            raise ValueError("NeuroKit error: signal_autocor(): The time lag exceeds the duration of the signal. ")
        else:
            r = r[lag]

    return r
