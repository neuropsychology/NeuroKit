# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


from .utils_phi import _phi
from .utils_phi import _phi_divide
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
        The embedding dimension (often denoted as 'm'), i.e., the length of compared runs of data. Typically 1, 2 or 3.
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
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> nk.entropy_sample(signal[0:100])
    """
    r = _get_r(signal, r=r)

    # nolds and Entropy implementation:
    phi = _phi(signal, order=order, r=r, metric='chebyshev', approximate=False)

    return _phi_divide(phi)
