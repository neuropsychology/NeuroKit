# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.stats
from pandas._libs.tslibs import BaseOffset

from .utils import _sanitize_multichannel


def entropy_shannon(signal, base=2):
    """Shannon entropy (SE)

    Python implementation of Shannon entropy (SE). Entropy is a measure of unpredictability of the state,
    or equivalently, of its average information content. Shannon entropy (SE) is one of the first and
    most basic measure of entropy and a foundational concept of information theory. Shannon’s entropy
    quantifies the amount of information in a variable.

    This function can be called either via ``entropy_shannon()`` or ``complexity_se()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    base: float
        The logarithmic base to use, defaults to 2. Note that ``scipy.stats.entropy``
        uses ``np.e`` as default (the natural logarithm).


    Returns
    ----------
    shanen : float
        The Shannon entropy of the single time series, or the mean ShEn
        across the channels of an n-dimensional time series.
    parameters : dict
        A dictionary containing additional information regarding the parameters used
        to compute Shannon entropy and the individual ShEn values of each
        channel if an n-dimensional time series is passed.

    See Also
    --------
    entropy_approximate, entropy_sample, entropy_fuzzy

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5, noise=0.1)
    >>> entropy, parameters = nk.entropy_shannon(signal)
    >>> entropy #doctest: +SKIP


    References
    -----------
    - `pyEntropy` <https://github.com/nikdon/pyEntropy>`_

    - `EntroPy` <https://github.com/raphaelvallat/entropy>`_

    - `nolds` <https://github.com/CSchoel/nolds>`_

    """
    # Initialize info dict
    info = {"Base": base}

    # Sanitize input
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        # n-dimensional
        signal = _sanitize_multichannel(signal)

        shen_values = []
        for i, colname in enumerate(signal):
            channel = np.array(signal[colname])
            shen = _entropy_shannon(channel)
            shen_values.append(shen)
        info["Values"] = shen_values
        out = np.mean(shen_values)

    else:
        # if one signal time series
        out = _entropy_shannon(signal)

    return out, info


def _entropy_shannon(signal, base=2):

    # Check if string
    if not isinstance(signal, str):
        signal = list(signal)

    return scipy.stats.entropy(pd.Series(signal).value_counts(), base=base)
