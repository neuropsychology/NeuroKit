import numpy


def hfd(X, Kmax):
    """ Compute Higuchi Fractal Dimension of a time series X. kmax
     is an HFD parameter
    """
    L = []
    x = []
    N = len(X)
    for k in range(1, Kmax):
        Lk = []
        for m in range(0, k):
            Lmk = 0
            for i in range(1, int(numpy.floor((N - m) / k))):
                Lmk += abs(X[m + i * k] - X[m + i * k - k])
            Lmk = Lmk * (N - 1) / numpy.floor((N - m) / float(k)) / k
            Lk.append(Lmk)
        L.append(numpy.log(numpy.mean(Lk)))
        x.append([numpy.log(float(1) / k), 1])

    (p, _, _, _) = numpy.linalg.lstsq(x, L)
    return p[0]


def pfd(X, D=None):
    """Compute Petrosian Fractal Dimension of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed using Numpy's difference function.

    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.
    """
    if D is None:
        D = numpy.diff(X)
        D = D.tolist()
    N_delta = 0  # number of sign changes in derivative of the signal
    for i in range(1, len(D)):
        if D[i] * D[i - 1] < 0:
            N_delta += 1
    n = len(X)
    return numpy.log10(n) / (
        numpy.log10(n) + numpy.log10(n / n + 0.4 * N_delta)
    )
