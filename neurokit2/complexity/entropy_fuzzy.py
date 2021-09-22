# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .utils import _get_r, _phi, _phi_divide, _sanitize_multichannel


def entropy_fuzzy(signal, delay=1, dimension=2, r="default", **kwargs):
    """Fuzzy entropy (FuzzyEn)

    Python implementations of the fuzzy entropy (FuzzyEn) of a signal.

    This function can be called either via ``entropy_fuzzy()`` or ``complexity_fuzzyen()``.

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
    **kwargs
        Other arguments.

    Returns
    ----------
    fuzzyen : float
        The fuzzy entropy of the single time series or the mean FuzzyEn
        across the channels of an n-dimensional time series.
    parameters : dict
        A dictionary containing additional information regarding the parameters used
        to compute fuzzy entropy and the individual FuzzyEn values of each
        channel if an n-dimensional time series is passed.

    See Also
    --------
    entropy_shannon, entropy_approximate, entropy_sample

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> entropy, parameters = nk.entropy_fuzzy(signal)
    >>> entropy #doctest: +SKIP

    """

    # Prepare parameters
    parameters = {'embedding_dimension': dimension,
                  'tau': delay}

    # sanitize input
    if signal.ndim > 1:
        # n-dimensional
        signal = _sanitize_multichannel(signal)

        fuzzyen_values = []
        for i, colname in enumerate(signal):
            channel = np.array(signal[colname])
            fuzzyen, tolerance = _entropy_fuzzy(channel, delay=delay, dimension=dimension,
                                                r=r, **kwargs)
            fuzzyen_values.append(fuzzyen)
        parameters['values'] = fuzzyen_values
        parameters['tolerance'] = tolerance
        out = np.mean(fuzzyen_values)

    else:
        # if one signal time series
        if isinstance(signal, (pd.Series)):
            signal = np.array(signal)
        out, parameters['tolerance'] = _entropy_fuzzy(signal, delay=delay, dimension=dimension,
                                                      r=r, **kwargs)

    return out, parameters


def _entropy_fuzzy(signal, delay=1, dimension=2, r="default", **kwargs):

    r = _get_r(signal, r=r, dimension=dimension)
    phi = _phi(signal, delay=delay, dimension=dimension, r=r, approximate=False, fuzzy=True, **kwargs)

    fuzzyen =  _phi_divide(phi)

    return fuzzyen, r
