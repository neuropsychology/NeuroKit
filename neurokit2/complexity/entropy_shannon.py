# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np
import pandas as pd
import scipy.stats

from ..misc import NeuroKitWarning
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
        The Shannon entropy of the single time series.
    info : dict
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
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        warn(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.",
            category=NeuroKitWarning,
        )

    # Check if string ('ABBA'), and convert each character to list (['A', 'B', 'B', 'A'])
    if not isinstance(signal, str):
        signal = list(signal)

    shanen = scipy.stats.entropy(pd.Series(signal).value_counts(), base=base)

    return shanen, {"Base": base}
