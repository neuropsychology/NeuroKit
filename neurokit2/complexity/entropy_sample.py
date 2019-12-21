# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


from .utils_embed import _embed




def _entropy_sample(signal, order=2, r="default", n=1, fuzzy=False):
    """
    Internal function adapted from https://github.com/ixjlyons/entro-py/blob/master/entropy.py
    """
    # Sanity checks
    signal = np.array(signal).astype(float)
    N = len(signal)

    if r == "default":
        r = 0.2 * np.std(signal, axis=-1, ddof=1)

    phi = [0, 0]  # phi(m), phi(m+1)
    for j in [0, 1]:
        m = order + j
        patterns = np.transpose(_embed(signal, m))

        if fuzzy:
            patterns -= np.mean(patterns, axis=0, keepdims=True)

        count = np.zeros(N-m)
        for i in range(N-m):
            if m == 1:
                sub = patterns[i]
            else:
                sub = patterns[:, [i]]
            dist = np.max(np.abs(patterns - sub), axis=0)

            if fuzzy:
                sim = np.exp(-np.power(dist, n) / r)
            else:
                sim = dist <= r

            count[i] = np.sum(sim) - 1

        phi[j] = np.mean(count) / (N-m-1)

    return np.log(phi[0] / phi[1])




def entropy_sample(signal, order=2, r="default"):
    """
    Calculate the sample entropy (SampEn) of a signal. Adapted from `entro-py <https://github.com/ixjlyons/entro-py/blob/master/entropy.py>`_.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    order : int
        The Embedding dimension (often denoted as 'm'), i.e., the length of compared run of data. Typically 1, 2 or 3.
    r : float
        Tolerance (i.e., filtering level - max absolute difference between segments). If 'default', will be set to 0.2 times the standard deviation of the signal.

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
    0.27503095489822205
    """
    return _entropy_sample(signal, order=order, r=r, n=1, fuzzy=False)