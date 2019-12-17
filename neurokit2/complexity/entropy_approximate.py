# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from .utils_phi import _phi

def entropy_approximate(signal, order=2, r="default"):
    """Compute the approximate entropy (ApEn).

    Approximate entropy is a technique used to quantify the amount of regularity and the unpredictability of fluctuations over time-series data. The advantages of ApEn include lower computational demand (ApEn can be designed to work for small data samples (< 50 data points) and can be applied in real tim) and less sensitive to noise. However, ApEn is heavily dependent on the record length and lacks relative consistency.


    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    order : int
        The Embedding dimension, i.e., the length of compared run of data. Typically 1, 2 or 3.
    r : int
        Filtering level. If 'default', will be set to 0.2 times the standard deviation of the signal.


    Returns
    ----------
    float
        The approximate entropy as float value.


    Example
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Download example EEG signal
    >>> signal = pd.read_csv('https://raw.github.com/neuropsychology/NeuroKit/master/data/example_eeg.txt', header=None)[0].values
    >>> nk.entropy_approximate(signal[0:100])
    0.38717434097485004


    References
    -----------
    - `EntroPy` <https://github.com/raphaelvallat/entropy>`_
    - Sabeti, M., Katebi, S., & Boostani, R. (2009). Entropy and complexity measures for EEG signal classification of schizophrenic and control participants. Artificial intelligence in medicine, 47(3), 263-274.
    """
    # Sanity checks
    if r == "default":
        r = 0.2 * np.std(signal, axis=-1, ddof=1)

    # Get phi
    phi = _phi(signal, order=order, r=r, metric='chebyshev', approximate=True)
    return(np.abs(np.subtract(phi[0], phi[1])))
