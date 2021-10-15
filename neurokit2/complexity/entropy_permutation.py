import numpy as np
import pandas as pd

from .complexity_embedding import complexity_embedding


def entropy_permutation(signal, dimension=3, delay=1, corrected=True):
    """Permutation Entropy (PEn).

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
    >>> pen, info = nk.entropy_permutation(signal, dimension=3, delay=1, corrected=False)

    """
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
    return pe, {"Corrected": corrected}
