import numpy as np


def signal_autocor(signal, normalize=True):
    """
    Auto-correlation of a 1-dimensional sequences.

    Parameters
    -----------
    signal : list, array or Series
        Vector of values.
    normalize : bool
        Normalize the autocorrelation output.

    Returns
    -------
    r
        The cross-correlation of the signal with itself at different time lags.
        Minimum time lag is 0, maximum time lag is the length of the signal.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> x = [1, 2, 3, 4, 5]
    >>> autocor = nk.signal_autocor(x)
    >>> autocor #doctest: +SKIP
    """
    r = np.correlate(signal, signal, mode='full')

    r = r[r.size // 2:]  # min time lag is 0

    if normalize is True:
        r = r / r[0]

    return r
