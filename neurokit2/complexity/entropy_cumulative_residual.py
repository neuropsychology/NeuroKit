import itertools

import numpy as np
import pandas as pd


def entropy_cumulative_residual(signal):
    """Cumulative residual entropy (CREn)

    The cumulative residual entropy is an alternative to the Shannon
    differential entropy with several advantageous properties, such as non-negativity.

    The implementation is based on
    `dit` <https://github.com/dit/dit/blob/master/dit/other/cumulative_residual_entropy.py>_.

    This function can be called either via ``entropy_cumulative_residual()`` or ``complexity_cren()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    Returns
    -------
    CREn : float
        The cumulative residual entropy.
    info : dict
        A dictionary containing 'Values' for each pair of events.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = [1, 2, 3, 4, 5, 6]
    >>> cren, info = nk.entropy_cumulative_residual(signal)
    >>> cren #doctest: +SKIP

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Check if string ('ABBA'), and convert each character to list (['A', 'B', 'B', 'A'])
    if not isinstance(signal, str):
        signal = list(signal)

    # Get probability of each event
    valscount = pd.Series(signal).value_counts(sort=True)
    events, probs = valscount.index.values, valscount.values / valscount.sum()

    cdf = {a: _ for a, _ in zip(events, np.cumsum(probs))}
    terms = np.zeros(len(events))
    for i, (a, b) in enumerate(_entropy_cumulative_residual_pairwise(events)):
        pgx = cdf[a]
        term = (b - a) * pgx * np.log2(pgx)
        terms[i] = term
    return -np.nansum(terms), {"Values": terms}


# =============================================================================
# Utilities
# =============================================================================
def _entropy_cumulative_residual_pairwise(events):
    pairs = itertools.tee(events, 2)
    pairs = list(zip(*pairs))
    for i, _ in enumerate(pairs[:-1]):
        pairs[i] = (pairs[i][0], pairs[i + 1][0])
    return pairs[:-1]
