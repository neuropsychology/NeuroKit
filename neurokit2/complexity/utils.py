# -*- coding: utf-8 -*-
import numpy as np
import sklearn.neighbors

from .embedding import embedding


def _get_embedded(signal, delay=1, dimension=2, r="default", metric='chebyshev', approximate=True):
    """
    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> delay = nk.embedding_delay(signal)
    >>>
    >>> embbededn, count = _get_embedded(signal,
                                         dimension,
                                         r= 0.2 * np.std(signal, ddof=1),
                                         delay=1,
                                         metric='chebyshev',
                                         approximate=False)
    """
    # Sanity checks
    if metric not in sklearn.neighbors.KDTree.valid_metrics:
        raise ValueError("NeuroKit error: _entropy_approximate_and_sample(): The given metric (%s) is not valid. The valid metric names are: %s" % (metric, sklearn.neighbors.KDTree.valid_metrics))

    # Get embedded
    embedded = embedding(signal, delay=delay, dimension=dimension)
    if approximate is False:
        embedded = embedded[:-1]  # Removes the last line

    # Get neighbors count
    count = sklearn.neighbors.KDTree(embedded, metric=metric).query_radius(embedded, r, count_only=True).astype(np.float64)

    return embedded, count




# =============================================================================
# Phi
# =============================================================================


def _phi(signal, delay=1, dimension=2, r="default", metric='chebyshev', approximate=True):
    """Common internal for `entropy_approximate` and `entropy_sample`.

    Adapted from `EntroPy <https://github.com/raphaelvallat/entropy>`_, check it out!
    """
    # Initialize phi
    phi = np.zeros(2)

    embedded1, count1 = _get_embedded(signal, delay, dimension, r, metric=metric, approximate=approximate)
    embedded2, count2 = _get_embedded(signal, delay, dimension + 1, r, metric=metric, approximate=True)

    if approximate is True:
        phi[0] = np.mean(np.log(count1 / embedded1.shape[0]))
        phi[1] = np.mean(np.log(count2 / embedded2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (embedded1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (embedded2.shape[0] - 1))
    return phi




def _phi_divide(phi):
    if phi[0] == 0:
        return -np.inf
    division = np.divide(phi[1], phi[0])
    if division == 0:
        return np.inf
    return -np.log(division)



# =============================================================================
# Get R
# =============================================================================

def _get_r(signal, r="default"):

    if isinstance(r, str) or r == None:
        r = 0.2 * np.std(signal, ddof=1)

    return r
