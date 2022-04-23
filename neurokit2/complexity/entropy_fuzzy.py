# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .entropy_approximate import entropy_approximate
from .entropy_sample import entropy_sample


def entropy_fuzzy(signal, delay=1, dimension=2, tolerance="sd", approximate=False, **kwargs):
    """**Fuzzy Entropy (FuzzyEn)**

    Computes the fuzzy entropy (FuzzyEn) of a signal.

    This function can be called either via ``entropy_fuzzy()`` or ``complexity_fuzzyen()``, or ``complexity_fuzzyapen()`` for its approximate version.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension()` to estimate the optimal value for this parameter.
    tolerance : float
        Tolerance (often denoted as *r*), distance to consider two data points as similar. If
        ``"sd"`` (default), will be set to :math:`0.2 * SD_{signal}`. See
        :func:`complexity_tolerance()` to estimate the optimal value for this parameter.
    **kwargs
        Other arguments.

    Returns
    ----------
    fuzzyen : float
        The fuzzy entropy of the single time series.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute fuzzy entropy.
    approximate : bool
        If ``True``, will compute the fuzzy approximate entropy (FuzzyApEn).

    See Also
    --------
    entropy_sample

    Examples
    ----------
    ..ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=5)

      fuzzyen, parameters = nk.entropy_fuzzy(signal)
      fuzzyen

      fuzzyapen, parameters = nk.entropy_fuzzy(signal, approximate=True)
      fuzzyapen

    """
    if approximate is False:
        out = entropy_sample(
            signal,
            delay=delay,
            dimension=dimension,
            tolerance=tolerance,
            fuzzy=True,
            **kwargs,
        )
    else:
        out = entropy_approximate(
            signal,
            delay=delay,
            dimension=dimension,
            tolerance=tolerance,
            fuzzy=True,
            **kwargs,
        )
    return out
