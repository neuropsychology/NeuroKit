# -*- coding: utf-8 -*-
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

from .embedding import embedding



def embedding_dimension(signal, delay=1, dimension_max=20, show=False, **kwargs):
    """Estimate optimal Dimension (m) for time-delay embedding

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau'). In practice, it is common to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
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

    E1, E2 = _embedding_dimension_afn(signal, dimension_seq, delay=1, show=False, **kwargs)

    if show is True:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

        ax1.plot(dimension_seq, E1)
        ax2.plot(dimension_seq, E2)

    return E2




# =============================================================================
# Internals
# =============================================================================
def _embedding_dimension_afn(signal, dimension_seq, delay=1, show=False, **kwargs):
    """
    """
    values = np.asarray([_embedding_dimension_afn_d(signal, dimension, delay, **kwargs) for dimension in dimension_seq]).T

    return values[0, :], values[1, :]

def _embedding_dimension_afn_d(signal, dimension, delay=1, R=10.0, A=2.0, metric='chebyshev', window=10, maxnum=None):
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

    # Compute the magnification and the increase in the near-neighbor
    # distances and return the averages.
    d = np.asarray([scipy.spatial.distance.chebyshev(i, j) for i, j in zip(y2, y2[index])])
    E = np.mean(d / dist)

    Es = np.mean(np.abs(y2[:, -1] - y2[index, -1]))
    return E, Es






def _embedding_dimension_neighbors(y, metric='chebyshev', window=0, maxnum=None):
    """Find nearest neighbors of all points in the given array.
    Finds the nearest neighbors of all points in the given array using
    SciPy's KDTree search.

    Parameters
    ----------
    y : ndarray
        N-dimensional array containing time-delayed vectors.
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
            dist, index = tree.query(x, k=k, p=p)
            valid = (np.abs(index - i) > window) & (dist > 0)

            if np.count_nonzero(valid):
                dists[i] = dist[valid][0]
                indices[i] = index[valid][0]
                break

            if k == (maxnum + 1):
                raise Exception('Could not find any near neighbor with a '
                                'nonzero distance.  Try increasing the '
                                'value of maxnum.')

    return np.squeeze(indices), np.squeeze(dists)
