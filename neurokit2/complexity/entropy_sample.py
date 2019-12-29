# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


from .utils_phi import _phi
from .utils_get_r import _get_r
from .entropy_fuzzy import _entropy_sample






def entropy_sample(signal, order=2, r="default"):
    """
    Calculate the sample entropy (SampEn) of a signal.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    order : int
        The Embedding dimension (often denoted as 'm'), i.e., the length of compared run of data. Typically 1, 2 or 3.
    r : float
        Tolerance (i.e., filtering level - max absolute difference between segments). If 'default', will be set to 0.2 times the standard deviation of the signal.

    See Also
    --------
    entropy_shannon, entropy_approximate, entropy_fuzzy

    Returns
    ----------
    float
        The sample entropy as float value.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=30, num=100))
    >>> nk.entropy_sample(signal[0:100])
    0.281711905465277
    """
    r = _get_r(signal, r=r)

    # entro-py implementation (https://github.com/ixjlyons/entro-py/blob/master/entropy.py):
#    phi = _entropy_sample(signal, order=order, r=r, n=1, fuzzy=False)

    # nolds and Entropy implementation:
    phi = _phi(signal, order=order, r=r, metric='chebyshev', approximate=False)

    return -np.log(np.divide(phi[1], phi[0]))
