# -*- coding: utf-8 -*-
import numpy as np
import sklearn.neighbors

from .embedding import embedding




def _get_embedded(signal, dimension, r, metric='chebyshev', approximate=True):
    """
    """
    if metric not in sklearn.neighbors.KDTree.valid_metrics:
        raise ValueError("NeuroKit error: _entropy_approximate_and_sample(): The given metric (%s) is not valid. The valid metric names are: %s" % (metric, sklearn.neighbors.KDTree.valid_metrics))


    embedded = embedding(signal, delay=1, dimension=dimension)
    if approximate is False:
        embedded = embedded[:-1]
    count = sklearn.neighbors.KDTree(embedded, metric=metric).query_radius(embedded, r, count_only=True).astype(np.float64)
    return embedded, count
