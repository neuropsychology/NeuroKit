# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .utils import _get_tolerance, _phi, _phi_divide


def entropy_sample(signal, delay=1, dimension=2, tolerance="default", **kwargs):
    """Sample Entropy (SampEn)

    Python implementation of the sample entropy (SampEn) of a signal. SampEn is a modification
    of ApEn used for assessing complexity of physiological time series signals. Mathematically,
    it is the negative natural logarithm of the conditional probability that two subseries
    similar for ``m`` points remain similar for ``m + 1``, where self-matches are
    not included in calculating the probability.

    This function can be called either via ``entropy_sample()`` or ``complexity_sampen()``.

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
    **kwargs : optional
        Other arguments.

    See Also
    --------
    entropy_shannon, entropy_approximate, entropy_fuzzy

    Returns
    ----------
    sampen : float
        The sample entropy of the single time series.
        If undefined conditional probabilities are detected (logarithm
        of sum of conditional probabilities is ``ln(0)``), ``np.inf`` will
        be returned, meaning it fails to retrieve 'accurate' regularity information.
        This tends to happen for short data segments, increasing tolerance
        levels might help avoid this.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute sample entropy.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> entropy, parameters = nk.entropy_sample(signal)
    >>> entropy #doctest: +SKIP

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Prepare parameters
    info = {"Dimension": dimension, "Delay": delay}

    info["Tolerance"] = _get_tolerance(signal, tolerance=tolerance, dimension=dimension)
    out = _entropy_sample(
        signal, tolerance=info["Tolerance"], delay=delay, dimension=dimension, **kwargs
    )

    return out, info


def _entropy_sample(signal, tolerance, delay=1, dimension=2, fuzzy=False, distance="chebyshev"):

    phi = _phi(
        signal,
        delay=delay,
        dimension=dimension,
        tolerance=tolerance,
        approximate=False,
        distance=distance,
        fuzzy=fuzzy,
    )
    sampen = _phi_divide(phi)

    return sampen
