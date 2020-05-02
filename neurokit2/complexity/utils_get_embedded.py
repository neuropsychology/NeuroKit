# -*- coding: utf-8 -*-
import numpy as np
import sklearn.neighbors

from .delay_embedding import delay_embedding




def _get_embedded(signal, order, r, metric='chebyshev', approximate=True):
    """
    """
    embedded = delay_embedding(signal, delay=1, dimension=dimension)
    if approximate is False:
        embedded = embedded[:-1]
    count = sklearn.neighbors.KDTree(embedded, metric=metric).query_radius(embedded, r, count_only=True).astype(np.float64)
    return embedded, count
