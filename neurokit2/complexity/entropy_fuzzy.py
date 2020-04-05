# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from .utils_embed import _embed
from .utils_get_r import _get_r
from .utils_phi import _phi_divide




def entropy_fuzzy(signal, order=2, r="default", n=1):
    """
    Calculate the fuzzy entropy (FuzzyEn) of a signal. Adapted from `entro-py <https://github.com/ixjlyons/entro-py/blob/master/entropy.py>`_.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    order : int
        The embedding dimension (often denoted as 'm'), i.e., the length of compared run of data. Typically 1, 2 or 3.
    r : float
        Tolerance (i.e., filtering level - max absolute difference between segments). If 'default', will be set to 0.2 times the standard deviation of the signal.
    n : float, optional
        Step width of fuzzy exponential function. Larger `n` makes the function
        more rectangular. Usually in the range 1-5 (default is 1).

    Returns
    ----------
    float
        The fuzzy entropy as float value.

    See Also
    --------
    entropy_shannon, entropy_approximate, entropy_sample

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> nk.entropy_fuzzy(signal)
    0.08481168552031555
    """
    r = _get_r(signal, r=r)
    phi = _entropy_sample(signal, order=order, r=r, n=1, fuzzy=True)

    return _phi_divide(phi)




# =============================================================================
# Internal
# =============================================================================

def _entropy_sample(signal, order=2, r="default", n=1, fuzzy=False):
    """
    Internal function adapted from https://github.com/ixjlyons/entro-py/blob/master/entropy.py
    With fixes (https://github.com/ixjlyons/entro-py/pull/2/files) by @CSchoel
    """
    # Sanity checks
    signal = np.array(signal).astype(float)
    N = len(signal)

    phi = [0, 0]  # phi(m), phi(m+1)
    for j in [0, 1]:
        m = order + j
        npat = N - order  # https://github.com/ixjlyons/entro-py/pull/2
#        patterns = np.transpose(_embed(signal, m))
        patterns = np.transpose(_embed(signal, m))[:, :npat]

        if fuzzy:
            patterns -= np.mean(patterns, axis=0, keepdims=True)

#        count = np.zeros(N-m)  # https://github.com/ixjlyons/entro-py/pull/2
#        for i in range(N-m):  # https://github.com/ixjlyons/entro-py/pull/2
        count = np.zeros(npat)
        for i in range(npat):
            if m == 1:
                sub = patterns[i]
            else:
                sub = patterns[:, [i]]
            dist = np.max(np.abs(patterns - sub), axis=0)

            if fuzzy:
                sim = np.exp(-np.power(dist, n) / r)
            else:
                sim = dist < r  # https://github.com/ixjlyons/entro-py/pull/2

            count[i] = np.sum(sim) - 1

#        phi[j] = np.mean(count) / (N-m-1)
        phi[j] = np.mean(count) / (N-order-1)  # https://github.com/ixjlyons/entro-py/pull/2

    return phi
