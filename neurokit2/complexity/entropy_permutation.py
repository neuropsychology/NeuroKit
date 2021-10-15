import numpy as np
import pandas as pd

from .complexity_embedding import complexity_embedding
from .utils import _get_coarsegrained, _get_scale


def entropy_permutation(signal, dimension=3, delay=1, corrected=True, scale=None):
    """Permutation Entropy (PEn) and Multiscale Permutation Entropy.

    Permutation Entropy (PE or PEn) is a robust measure of the complexity of a dynamic system by
    capturing the order relations between values of a time series and extracting a probability
    distribution of the ordinal patterns (see Henry and Judge, 2019). This implementation is based on
    `pyEntropy <https://github.com/nikdon/pyEntropy>`_.

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
    corrected : bool
        If True, divide by log2(factorial(m)) to normalize the entropy
        between 0 and 1. Otherwise, return the permutation entropy in bit.

    Returns
    -------
    PEn : float
        Permutation Entropy

    References
    ----------
    - https://github.com/nikdon/pyEntropy
    - Zanin, M., Zunino, L., Rosso, O. A., & Papo, D. (2012). Permutation entropy and its main biomedical and econophysics applications: a review. Entropy, 14(8), 1553-1577.
    - Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time series. Physical review letters, 88(17), 174102.


    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, sampling_rate=100, frequency=[5, 6], noise=0.5)
    >>>
    >>> # Permutation Entropy
    >>> pen, info = nk.entropy_permutation(signal, dimension=3, delay=1, corrected=False)
    >>> pen
    >>> # Multiscale Permutation Entropy
    >>> pen, info = nk.entropy_permutation(signal, dimension=3, scale = "default")

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    info = {"Corrected": corrected, "Scale": None}

    # Multiscale
    if scale is not None:
        info["Scale"] = _get_scale(signal, scale=scale, dimension=dimension)
        info["Values"] = np.full(len(info["Scale"]), np.nan)
        for i, tau in enumerate(info["Scale"]):
            y = _get_coarsegrained(signal, tau)
            info["Values"][i] = _entropy_permutation(
                y, delay=1, dimension=dimension, corrected=corrected
            )
        # Remove inf, nan and 0
        vals = info["Values"].copy()[~np.isnan(info["Values"])]
        vals = vals[vals != np.inf]
        vals = vals[vals != -np.inf]

        # The MSE index is quantified as the area under the curve (AUC),
        # which is like the sum normalized by the number of values. It's similar to the mean.
        pe = np.trapz(vals) / len(vals)

    # Regular
    else:
        pe = _entropy_permutation(signal, dimension=dimension, delay=delay, corrected=corrected)
    return pe, info


# =============================================================================
# Internal
# =============================================================================
def _entropy_permutation(signal, dimension=3, delay=1, corrected=True):
    # Embed x and sort the order of permutations
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension).argsort(
        kind="quicksort"
    )
    # Associate unique integer to each permutations
    multiplier = np.power(dimension, np.arange(dimension))
    values = (np.multiply(embedded, multiplier)).sum(1)
    # Return the counts
    _, c = np.unique(values, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if corrected:
        pe /= np.log2(np.math.factorial(dimension))
    return pe
