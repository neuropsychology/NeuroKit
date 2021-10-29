# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .utils import _get_tolerance, _phi, _phi_divide


def entropy_fuzzy(signal, delay=1, dimension=2, tolerance="default", **kwargs):
    """Fuzzy entropy (FuzzyEn)

    Python implementations of the fuzzy entropy (FuzzyEn) of a signal.

    This function can be called either via ``entropy_fuzzy()`` or ``complexity_fuzzyen()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common
        to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or
        to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns
        an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    tolerance : float
        Tolerance (often denoted as 'r', i.e., filtering level - max absolute difference between segments).
        If 'default', will be set to 0.2 times the standard deviation of the signal (for dimension = 2).
    **kwargs
        Other arguments.

    Returns
    ----------
    fuzzyen : float
        The fuzzy entropy of the single time series.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute fuzzy entropy.

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

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Prepare parameters
    info = {'Dimension': dimension,
            'Delay': delay}

    info["Tolerance"] = _get_tolerance(signal, tolerance=tolerance, dimension=dimension)
    out = _entropy_fuzzy(signal, tolerance=info["Tolerance"], delay=delay, dimension=dimension,
                         **kwargs)

    return out, info


def _entropy_fuzzy(signal, tolerance, delay=1, dimension=2, **kwargs):

    phi = _phi(signal, delay=delay, dimension=dimension, tolerance=tolerance, approximate=False, fuzzy=True, **kwargs)

    fuzzyen = _phi_divide(phi)

    return fuzzyen
