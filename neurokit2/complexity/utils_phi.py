# -*- coding: utf-8 -*-
import numpy as np
import sklearn.neighbors

from .utils_get_embedded import _get_embedded


def _phi(signal, dimension, r="default", metric='chebyshev', approximate=True):
    """Common internal for `entropy_approximate` and `entropy_sample`.

    Adapted from `EntroPy <https://github.com/raphaelvallat/entropy>`_, check it out!
    """
    # Sanity check
    if metric not in sklearn.neighbors.KDTree.valid_metrics:
        raise ValueError("NeuroKit error: _entropy_approximate_and_sample(): The given metric (%s) is not valid. The valid metric names are: %s" % (metric, sklearn.neighbors.KDTree.valid_metrics))

    # Initialize phi
    phi = np.zeros(2)

    # compute phi(dimension, r)
    embedded1, count1 = _get_embedded(signal, dimension, r, metric=metric, approximate=approximate)
    # compute phi(dimension + 1, r)
    embedded2, count2 = _get_embedded(signal, dimension + 1, r, metric=metric, approximate=True)

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
