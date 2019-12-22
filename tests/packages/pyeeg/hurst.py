import numpy


def hurst(X):
    """ Compute the Hurst exponent of X. If the output H=0.5,the behavior
    of the time-series is similar to random walk. If H<0.5, the time-series
    cover less "distance" than a random walk, vice verse.

    Parameters
    ----------

    X

        list

        a time series

    Returns
    -------
    H

        float

        Hurst exponent

    Notes
    --------
    Author of this function is Xin Liu

    Examples
    --------

    >>> import pyeeg
    >>> from numpy.random import randn
    >>> a = randn(4096)
    >>> pyeeg.hurst(a)
    0.5057444

    """
    X = numpy.array(X)
    N = X.size
    T = numpy.arange(1, N + 1)
    Y = numpy.cumsum(X)
    Ave_T = Y / T

    S_T = numpy.zeros(N)
    R_T = numpy.zeros(N)

    for i in range(N):
        S_T[i] = numpy.std(X[:i + 1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = numpy.ptp(X_T[:i + 1])

    R_S = R_T / S_T
    R_S = numpy.log(R_S)[1:]
    n = numpy.log(T)[1:]
    A = numpy.column_stack((n, numpy.ones(n.size)))
    [m, c] = numpy.linalg.lstsq(A, R_S)[0]
    H = m
    return H
