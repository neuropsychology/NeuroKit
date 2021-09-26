# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .utils import _get_r, _phi, _phi_divide, _sanitize_multichannel


def entropy_sample(signal, delay=1, dimension=2, r="default", **kwargs):
    """Sample Entropy (SampEn)

    Python implementation of the sample entropy (SampEn) of a signal.

    This function can be called either via ``entropy_sample()`` or ``complexity_sampen()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series, np.ndarray, pd.DataFrame]
        The signal (i.e., a time series) in the form of a vector of values or in
        the form of an n-dimensional array (with a shape of len(channels) x len(samples))
        or dataframe.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common
        to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or
        to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns
        an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    r : float
        Tolerance (i.e., filtering level - max absolute difference between segments). If 'default',
        will be set to 0.2 times the standard deviation of the signal (for dimension = 2).
    **kwargs : optional
        Other arguments.

    See Also
    --------
    entropy_shannon, entropy_approximate, entropy_fuzzy

    Returns
    ----------
    sampen : float
        The sample entropy of the single time series or the mean SampEn
        across the channels of an n-dimensional time series.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute sample entropy and the individual SampEn values of each
        channel if an n-dimensional time series is passed.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> entropy, parameters = nk.entropy_sample(signal)
    >>> entropy #doctest: +SKIP

    """
    # Prepare parameters
    info = {'Dimension': dimension,
            'Tau': delay}

    # Sanitize input
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        # n-dimensional
        signal = _sanitize_multichannel(signal)
        # Get tolerance
        info["Tolerance"] = _get_r(signal, r=r, dimension=dimension)
        info["Values"] = np.full(signal.shape[1], np.nan)  # Initialize empty vector of values
        for i, colname in enumerate(signal):
            info["Values"][i] = _entropy_sample(signal[colname], r=info["Tolerance"],
                                                delay=delay, dimension=dimension, **kwargs)
        out = np.mean(info["Values"])
    else:
        # one single time series
        info["Tolerance"] = _get_r(signal, r=r, dimension=dimension)
        out = _entropy_sample(signal, r=info["Tolerance"], delay=delay, dimension=dimension, **kwargs)

    return out, info


def _entropy_sample(signal, r, delay=1, dimension=2, **kwargs):

    phi = _phi(signal, delay=delay, dimension=dimension, r=r, approximate=False, **kwargs)
    sampen = _phi_divide(phi)

    return sampen
