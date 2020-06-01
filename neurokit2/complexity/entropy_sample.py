# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .utils import _get_r, _phi, _phi_divide


def entropy_sample(signal, delay=1, dimension=2, r="default", **kwargs):
    """
    Sample Entropy (SampEn)

    Python implementation of the sample entropy (SampEn) of a signal.

    This function can be called either via ``entropy_sample()`` or ``complexity_sampen()``.

    Parameters
    ----------
    signal : list, array or Series
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically 2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    r : float
        Tolerance (i.e., filtering level - max absolute difference between segments). If 'default', will be set to 0.2 times the standard deviation of the signal (for dimension = 2).

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
    >>> entropy = nk.entropy_sample(signal)
    >>> entropy #doctest: +SKIP

    """
    r = _get_r(signal, r=r, dimension=dimension)
    phi = _phi(signal, delay=delay, dimension=dimension, r=r, approximate=False, **kwargs)

    return _phi_divide(phi)
