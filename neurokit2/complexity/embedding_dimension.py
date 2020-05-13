# -*- coding: utf-8 -*-
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

from .embedding import embedding



def embedding_dimension(signal, delay=1, dimension_max=20, method="afnn", show=False, **kwargs):
    """Estimate optimal Dimension (m) for time-delay embedding
    El(d) = E(d + 1)/E(d). E1(d) stops changing when d is greater
    than some value d0 if the time series comes from an attractor. Then d0 + 1
    is the minimum embedding dimension we look for.
    E2(d) = E*(d + 1)/E*(d). E2(d) is a useful quantity to distinguish
    deterministic signals from stochastic signals. For random data, since the
    future values are independent of the past values, E2(d) will be equal to 1
    for any d. For deterministic data, E2(d) is certainly related to d, it
    cannot be a constant for all d; there must exist somed's such that E2(d)
    is not 1.

    Parameters
    ----------
    signal : list, array or Series
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension_max : int
        The maximum embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order') to test.
    show : bool
        Visualize the result.

    Returns
    -------
    int
        Optimal dimension.

    See Also
    ------------
    embedding_delay, embedding

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> # Artifical example
    >>> signal = nk.signal_simulate(duration=10, frequency=1, noise=0.01)
    >>> delay = nk.embedding_delay(signal, delay_max=500)
    >>>
    >>> values = nk.embedding_dimension(signal, delay=delay, dimension_max=20, show=True)
    >>>
    >>> # Realistic example
    >>> ecg = nk.ecg_simulate(duration=60*6, sampling_rate=150)
    >>> signal = nk.ecg_rate(nk.ecg_peaks(ecg, sampling_rate=150)[0], sampling_rate=150)
    >>> delay = nk.embedding_delay(signal, delay_max=300)
    >>>
    >>> # This doesn't work for some reasons
    >>> # values = nk.embedding_dimension(signal, delay=delay, dimension_max=20, show=True)

    References
    -----------
    - Cao, L. (1997). Practical method for determining the minimum embedding dimension of a scalar time series. Physica D: Nonlinear Phenomena, 110(1-2), 43-50.
    """
    # Initalize vectors
    if isinstance(dimension_max, int):
        dimension_seq = np.arange(1, dimension_max + 1)
    else:
        dimension_seq = np.array(dimension_max)

    # Method
    method = method.lower()
    if method in ["afnn"]:
        E, Es = _embedding_dimension_afn(signal, dimension_seq=dimension_seq, delay=delay, show=show, **kwargs)
        E1 = E[1:] / E[:-1]
        E2 = Es[1:] / Es[:-1]

    if show is True:
        plt.title(r'AFN')
        plt.xlabel(r'Embedding dimension $d$')
        plt.ylabel(r'$E_1(d)$ and $E_2(d)$')
        plt.plot(dimension_seq[:-1], E1, 'bo-', label=r'$E_1(d)$')
        plt.plot(dimension_seq[:-1], E2, 'go-', label=r'$E_2(d)$')
        plt.legend()

    # To find where E1 saturates, set a threshold of difference
#    threshold = 0.1 * (np.max(E1) - np.min(E1))
    min_dimension = [i for i, x in enumerate(E1 >= 0.8 * np.max(E1)) if x][0] + 1

    return min_dimension




# =============================================================================
# Methods
# =============================================================================
def _embedding_dimension_afn(signal, dimension_seq, delay=1, show=False, **kwargs):
    """Return E(d) and E^*(d) for a all d in dimension_seq.
    E(d) and E^*(d) will be used to calculate E1(d) and E2(d)
    """
    values = np.asarray([_embedding_dimension_afn_d(signal, dimension, delay, **kwargs) for dimension in dimension_seq]).T
    E, Es = values[0, :], values[1, :]

    return E, Es


def _embedding_dimension_afn_d(signal, dimension, delay=1, metric='chebyshev', window=10, maxnum=None):
    """Return E(d) and E^*(d) for a single d.
    Returns E(d) and E^*(d) for the AFN method for a single d.
    """
    # We need to reduce the number of points in dimension d by tau
    # so that after reconstruction, there'll be equal number of points
    # at both dimension d as well as dimension d + 1.
    y1 = embedding(signal[:-delay], delay=delay, dimension=dimension)
    y2 = embedding(signal, delay=delay, dimension=dimension + 1)

    # Find near neighbors in dimension d.
    index, dist = _embedding_dimension_neighbors(y1, metric=metric, window=window, maxnum=maxnum)

    # Compute the near-neighbor distances in d + 1 dimension
    d = np.asarray([scipy.spatial.distance.chebyshev(i, j) for i, j in zip(y2, y2[index])])
    # Compute the ratio of near-neighbor distances in d + 1 over d dimension
    # Its average is E(d)
    E = np.mean(d / dist)

    # Calculate E^*(d)
    Es = np.mean(np.abs(y2[:, -1] - y2[index, -1]))
    return E, Es

def _embedding_dimension_afn_d(signal, dimension, delay=1, R=10.0, A=2.0, metric='euclidean', window=10, maxnum=None):
    """Return fraction of false nearest neighbors for a single d.
    """
    # We need to reduce the number of points in dimension d by tau
    # so that after reconstruction, there'll be equal number of points
    # at both dimension d as well as dimension d + 1.
    y1 = embedding(signal[:-delay], delay=delay, dimension=dimension)
    y2 = embedding(signal, delay=delay, dimension=dimension + 1)

    # Find near neighbors in dimension d.
    index, dist = _embedding_dimension_neighbors(y1, metric=metric, window=window, maxnum=maxnum)
    # Compute the near-neighbor distances in d + 1 dimension
    d = np.asarray([scipy.spatial.distance.chebyshev(i, j) for i, j in zip(y2, y2[index])])

    # Find all potential false neighbors using Kennel et al.'s tests.
    f1 = np.mean(np.abs(y2[:, -1] - y2[index, -1]) / dist > R)
    f2 = np.mean(d / np.std(signal) > A)
    f3 = np.mean(f1 | f2)

    return f1, f2, f3






def _embedding_dimension_neighbors(signal, dimension_max=20, delay=1, metric='chebyshev', window=0, maxnum=None, show=False):
    """Find nearest neighbors of all points in the given array.
    Finds the nearest neighbors of all points in the given array using
    SciPy's KDTree search.

    Parameters
    ----------
    signal : ndarray, array, list or Series
        embedded signal: N-dimensional array containing time-delayed vectors,
        or
        signal: 1-D array (e.g.time series) of signal in the form of a vector
        of values. If signal is input, embedded signal will be created using
        the input dimension and delay.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In
        practice, it is common to have a fixed time lag (corresponding for
        instance to the sampling rate; Gautama, 2003), or to find a suitable
        value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension_max : int
        The maximum embedding dimension (often denoted 'm' or 'd', sometimes
        referred to as 'order') to test.
    metric : string, optional (default = 'chebyshev')
        Metric to use for distance computation.  Must be one of
        "cityblock" (aka the Manhattan metric), "chebyshev" (aka the
        maximum norm metric), or "euclidean".
    window : int, optional (default = 0)
        Minimum temporal separation (Theiler window) that should exist
        between near neighbors.  This is crucial while computing
        Lyapunov exponents and the correlation dimension.
    maxnum : int, optional (default = None (optimum))
        Maximum number of near neighbors that should be found for each
        point.  In rare cases, when there are no neighbors that are at a
        nonzero distance, this will have to be increased (i.e., beyond
        2 * window + 3).

    Returns
    -------
    index : array
        Array containing indices of near neighbors.
    dist : array
        Array containing near neighbor distances.
    """

    # Sanity checks
    if len(signal.shape) == 1:
        y = embedding(signal, delay=delay, dimension=dimension_max)
    else:
        y = signal

    if metric == 'cityblock':
        p = 1
    elif metric == 'euclidean':
        p = 2
    elif metric == 'chebyshev':
        p = np.inf
    else:
        raise ValueError('Unknown metric.  Should be one of "cityblock", '
                         '"euclidean", or "chebyshev".')

    tree = scipy.spatial.cKDTree(y)
    n = len(y)

    if not maxnum:
        maxnum = (window + 1) + 1 + (window + 1)
    else:
        maxnum = max(1, maxnum)

    if maxnum >= n:
        raise ValueError('maxnum is bigger than array length.')

    dists = np.empty(n)
    indices = np.empty(n, dtype=int)

    for i, x in enumerate(y):
        for k in range(2, maxnum + 2):
            # querry for k number of nearest neighbours
            dist, index = tree.query(x, k=k, p=p)
            # remove points that are closer than min temporal separation
            # remove self reference (d > 0)
            valid = (np.abs(index - i) > window) & (dist > 0)

            if np.count_nonzero(valid):
                dists[i] = dist[valid][0]
                indices[i] = index[valid][0]
                break

            if k == (maxnum + 1):
                raise Exception('Could not find any near neighbor with a '
                                'nonzero distance.  Try increasing the '
                                'value of maxnum.')

    indices, values = np.squeeze(indices), np.squeeze(dists)

    if show is True:
        plt.plot(indices, values)

    return indices, values
