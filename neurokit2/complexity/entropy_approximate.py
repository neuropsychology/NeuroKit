# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from .utils import _phi, _get_r



def entropy_approximate(signal, delay=1, dimension=2, r="default", **kwargs):
    """Approximate entropy (ApEn)

    Python implementations of the approximate entropy (ApEn). Approximate entropy is a technique used to quantify the amount of regularity and the unpredictability of fluctuations over time-series data. The advantages of ApEn include lower computational demand (ApEn can be designed to work for small data samples (< 50 data points) and can be applied in real tim) and less sensitive to noise. However, ApEn is heavily dependent on the record length and lacks relative consistency.

    This function can be called either via ``entropy_approximate()`` or ``complexity_apen()``.


    Parameters
    ----------
    signal : list, array or Series
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically 2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    r : float
        Tolerance (similarity threshold). It corresponds to the filtering level - max absolute difference between segments. If 'default', will be set to 0.2 times the standard deviation of the signal (for dimension = 2).

    See Also
    --------
    entropy_shannon, entropy_sample, entropy_fuzzy

    Returns
    ----------
    float
        The approximate entropy as float value.


    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> nk.entropy_approximate(signal)
    0.08837414074679684


    References
    -----------
    - `EntroPy` <https://github.com/raphaelvallat/entropy>`_
    - Sabeti, M., Katebi, S., & Boostani, R. (2009). Entropy and complexity measures for EEG signal classification of schizophrenic and control participants. Artificial intelligence in medicine, 47(3), 263-274.
    """
    r = _get_r(signal, r=r, dimension=dimension)

    # Get phi
    phi = _phi(signal, delay=delay, dimension=dimension, r=r, approximate=True, **kwargs)
    return np.abs(np.subtract(phi[0], phi[1]))
