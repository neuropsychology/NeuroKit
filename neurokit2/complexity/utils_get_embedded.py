# -*- coding: utf-8 -*-
import numpy as np
import sklearn.neighbors

from .utils_embed import _embed




def _get_embedded(signal, order, r, metric='chebyshev', approximate=True):
    """
    """
    embedded = _embed(signal, order, 1)
    if approximate is False:
        embedded = embedded[:-1]
    count = sklearn.neighbors.KDTree(embedded, metric=metric).query_radius(embedded, r, count_only=True).astype(np.float64)
    return embedded, count
