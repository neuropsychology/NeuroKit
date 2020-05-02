# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from .utils_phi import _phi
from .utils_get_r import _get_r



def entropy_approximate(signal, dimension=2, r="default"):
    """Compute the approximate entropy (ApEn).

    Approximate entropy is a technique used to quantify the amount of regularity and the unpredictability of fluctuations over time-series data. The advantages of ApEn include lower computational demand (ApEn can be designed to work for small data samples (< 50 data points) and can be applied in real tim) and less sensitive to noise. However, ApEn is heavily dependent on the record length and lacks relative consistency.


    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically 2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    r : float
        Tolerance (i.e., filtering level - max absolute difference between segments). If 'default', will be set to 0.2 times the standard deviation of the signal.

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
    >>> nk.entropy_approximate(signal[0:100])
    0.2227235476697098


    References
    -----------
    - `EntroPy` <https://github.com/raphaelvallat/entropy>`_
    - Sabeti, M., Katebi, S., & Boostani, R. (2009). Entropy and complexity measures for EEG signal classification of schizophrenic and control participants. Artificial intelligence in medicine, 47(3), 263-274.
    """
    r = _get_r(signal, r=r)

    # Get phi
    phi = _phi(signal, dimension=dimension, r=r, metric='chebyshev', approximate=True)
    return(np.abs(np.subtract(phi[0], phi[1])))
