# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

from .complexity_delay import embedding_delay
from .complexity_dimension import embedding_dimension
from .entropy_approximate import entropy_approximate


def complexity_r(signal, delay=None, dimension=None, default=False):
    """Estimate optimal tolerance (similarity threshold)
    Parameters
    ----------
    signal : list, array or Series
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically 2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    default : bool
        If 'default', r will be set to 0.2 times the standard deviation of the signal.

    Returns
    ----------
    float
        The optimal r as float value.


    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> delay = nk.embedding_delay(signal, delay_max=100, method="fraser1986")
    >>> dimension = nk.embedding_dimension(signal, delay=delay, dimension_max=20)
    >>> nk.complexity_r(signal, delay, dimension)
    0.010609254363011076

olerance (similarity threshold). It corresponds to the filtering level - max absolute difference between segments.

    References
    -----------
    - `Lu, S., Chen, X., Kanters, J. K., Solomon, I. C., & Chon, K. H. (2008). Automatic selection of the threshold value $ r $ for approximate entropy. IEEE Transactions on Biomedical Engineering, 55(8), 1966-1972.
    """

    if not delay:
        delay = embedding_delay(signal, delay_max=100, method="fraser1986")
    if not dimension:
        dimension = embedding_dimension(signal, delay=delay, dimension_max=20, show=True)

    modulator = np.arange(0.02, 0.5, 0.01)
    r_range = modulator * np.std(signal, ddof=1)

    ApEn = np.zeros_like(r_range)

    for i, r in enumerate(r_range):
        ApEn[i] = entropy_approximate(signal, delay=delay, dimension=dimension, r=r_range[i])

    r = r_range[np.argmax(ApEn)]

    return r
