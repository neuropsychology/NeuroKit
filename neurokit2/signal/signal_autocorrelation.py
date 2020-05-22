import numpy as np

def signal_autocor(x, normalize=True):
    """
    Auto-correlation of a 1-dimensional sequences.

    Parameters
    -----------
    x : list, array or Series
        Vector of values.
    normalize : bool
        Normalize the autocorrelation output.

    Returns
    -------
    r
        The cross-correlation of x with itself at different time lags.
        Minimum time lag is 0, maximum time lag is the length of x.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> x = [1, 2, 3, 4, 5]
    >>> autocor = nk.signal_autocor(x)
    >>> autocor #doctest: +SKIP
    """
    r = np.correlate(x, x, mode='full')

    r = r[r.size // 2:]  # min time lag is 0

    if normalize is True:
        r = r / r[0]

    return r
