# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


from .entropy_sample import _entropy_sample



def entropy_fuzzy(signal, order=2, r="default", n=1):
    """
    Calculate the fuzzy entropy (FuzzyEn) of a signal. Adapted from `entro-py <https://github.com/ixjlyons/entro-py/blob/master/entropy.py>`_.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    order : int
        The Embedding dimension (often denoted as 'm'), i.e., the length of compared run of data. Typically 1, 2 or 3.
    r : float
        Tolerance (i.e., filtering level - max absolute difference between segments). If 'default', will be set to 0.2 times the standard deviation of the signal.
    n : float, optional
        Step width of fuzzy exponential function. Larger `n` makes the function
        more rectangular. Usually in the range 1-5 (default is 1).

    Returns
    ----------
    float
        The fuzzy entropy as float value.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=30, num=100))
    >>> nk.entropy_fuzzy(signal[0:100])
    0.27492692805526253
    """
    return _entropy_sample(signal, order=order, r=r, n=1, fuzzy=True)