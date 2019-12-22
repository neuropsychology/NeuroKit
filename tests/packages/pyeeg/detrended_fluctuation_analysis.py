from __future__ import print_function

import numpy


def dfa(X, Ave=None, L=None):
    """Compute Detrended Fluctuation Analysis from a time series X and length of
    boxes L.

    The first step to compute DFA is to integrate the signal. Let original
    series be X= [x(1), x(2), ..., x(N)].

    The integrated signal Y = [y(1), y(2), ..., y(N)] is obtained as follows
    y(k) = \sum_{i=1}^{k}{x(i)-Ave} where Ave is the mean of X.

    The second step is to partition/slice/segment the integrated sequence Y
    into boxes. At least two boxes are needed for computing DFA. Box sizes are
    specified by the L argument of this function. By default, it is from 1/5 of
    signal length to one (x-5)-th of the signal length, where x is the nearest
    power of 2 from the length of the signal, i.e., 1/16, 1/32, 1/64, 1/128,
    ...

    In each box, a linear least square fitting is employed on data in the box.
    Denote the series on fitted line as Yn. Its k-th elements, yn(k),
    corresponds to y(k).

    For fitting in each box, there is a residue, the sum of squares of all
    offsets, difference between actual points and points on fitted line.

    F(n) denotes the square root of average total residue in all boxes when box
    length is n, thus
    Total_Residue = \sum_{k=1}^{N}{(y(k)-yn(k))}
    F(n) = \sqrt(Total_Residue/N)

    The computing to F(n) is carried out for every box length n. Therefore, a
    relationship between n and F(n) can be obtained. In general, F(n) increases
    when n increases.

    Finally, the relationship between F(n) and n is analyzed. A least square
    fitting is performed between log(F(n)) and log(n). The slope of the fitting
    line is the DFA value, denoted as Alpha. To white noise, Alpha should be
    0.5. Higher level of signal complexity is related to higher Alpha.

    Parameters
    ----------

    X:
        1-D Python list or numpy array
        a time series

    Ave:
        integer, optional
        The average value of the time series

    L:
        1-D Python list of integers
        A list of box size, integers in ascending order

    Returns
    -------

    Alpha:
        integer
        the result of DFA analysis, thus the slope of fitting line of log(F(n))
        vs. log(n). where n is the

    Examples
    --------
    >>> import pyeeg
    >>> numpy.random.seed(2018)
    >>> print(pyeeg.dfa(numpy.random.randn(4096))) 
    0.5121794335479618

    Reference
    ---------
    Peng C-K, Havlin S, Stanley HE, Goldberger AL. Quantification of scaling
    exponents and crossover phenomena in nonstationary heartbeat time series.
    _Chaos_ 1995;5:82-87

    Notes
    -----

    This value depends on the box sizes very much. When the input is a white
    noise, this value should be 0.5. But, some choices on box sizes can lead to
    the value lower or higher than 0.5, e.g. 0.38 or 0.58.

    Based on many test, I set the box sizes from 1/5 of    signal length to one
    (x-5)-th of the signal length, where x is the nearest power of 2 from the
    length of the signal, i.e., 1/16, 1/32, 1/64, 1/128, ...

    You may generate a list of box sizes and pass in such a list as a
    parameter.

    """

    X = numpy.array(X)

    if Ave is None:
        Ave = numpy.mean(X)

    Y = numpy.cumsum(X)
    Y -= Ave

    if L is None:
        L = numpy.floor(len(X) * 1 / (
            2 ** numpy.array(list(range(4, int(numpy.log2(len(X))) - 4))))
        )

    F = numpy.zeros(len(L))  # F(n) of different given box length n

    for i in range(0, len(L)):
        n = int(L[i])                        # for each box length L[i]
        if n == 0:
            print("time series is too short while the box length is too big")
            print("abort")
            exit()
        for j in range(0, len(X), n):  # for each box
            if j + n < len(X):
                c = list(range(j, j + n))
                # coordinates of time in the box
                c = numpy.vstack([c, numpy.ones(n)]).T
                # the value of data in the box
                y = Y[j:j + n]
                # add residue in this box
                F[i] += numpy.linalg.lstsq(c, y)[1]
        F[i] /= ((len(X) / n) * n)
    F = numpy.sqrt(F)

    Alpha = numpy.linalg.lstsq(numpy.vstack(
        [numpy.log(L), numpy.ones(len(L))]
    ).T, numpy.log(F))[0][0]

    return Alpha



if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True) 
