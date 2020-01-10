# -*- coding: utf-8 -*-
import numpy as np
import sklearn.neighbors

from .utils_embed import _embed




def _get_embedded(signal, order, r, metric='chebyshev', approximate=True):
    embedded = _embed(signal, order, 1)
    if approximate is False:
        embedded = embedded[:-1]
    count = sklearn.neighbors.KDTree(embedded, metric=metric).query_radius(embedded, r, count_only=True).astype(np.float64)
    return embedded, count






def _phi(signal, order, r="default", metric='chebyshev', approximate=True):
    """Common internal for `entropy_approximate` and `entropy_sample`.

    Adapted from `EntroPy <https://github.com/raphaelvallat/entropy>`_, check it out!
    """
    # Sanity check
    if metric not in sklearn.neighbors.KDTree.valid_metrics:
        raise ValueError("NeuroKit error: _entropy_approximate_and_sample(): The given metric (%s) is not valid. The valid metric names are: %s" % (metric, sklearn.neighbors.KDTree.valid_metrics))

    # Initialize phi
    phi = np.zeros(2)

    # compute phi(order, r)
    embedded1, count1 = _get_embedded(signal, order, r, metric=metric, approximate=approximate)
    # compute phi(order + 1, r)
    embedded2, count2 = _get_embedded(signal, order + 1, r, metric=metric, approximate=True)

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