# -*- coding: utf-8 -*-
#import numpy as np
#
#from .utils_embed import _embed
#
#
#def complexity_correlation(signal):
#    """
#    Computes Detrended Fluctuation Analysis (DFA) on the time series data.
#
#    Parameters
#    ----------
#    signal : list, array or Series
#        The signal channel in the form of a vector of values.
#    windows : list
#        The length of the data in each subseries. Defaults to None.
#    overlap : bool
#        Defaults to True, where the windows will have a 50% overlap
#        with each other, otherwise non-overlapping windows will be used.
#    order : int
#        The order of the trend.
#
#    Returns
#    ----------
#    poly : float
#        The estimate alpha of the Hurst parameter.
#
#    Examples
#    ----------
#    >>> import neurokit2 as nk
#    >>>
#    >>> signal = nk.signal_simulate(duration=2, frequency=5)
#    >>> nk.complexity_dfa(signal)
#
#
#    References
#    -----------
#    - `nolds` <https://github.com/CSchoel/nolds/blob/master/nolds/measures.py>
#    """
#    # Sanity-check input
#    signal = np.asarray(signal)
#    N = len(signal)
#
#    orbit = _embed(data, emb_dim, lag=1)
##    embedded2, count2 = _get_embedded(signal, order + 1, r, metric=metric, approximate=True)
#    pass
