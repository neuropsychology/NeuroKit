import itertools

import numpy as np
import pandas as pd

from .fractal_petrosian import _complexity_binarize


def entropy_cumulative_residual(signal, method=None, show=False):
    """**Cumulative residual entropy (CREn)**

    The cumulative residual entropy is an alternative to the Shannon
    differential entropy with several advantageous properties, such as non-negativity.

    Similarly to :func:`Shannon entropy <entropy_shannon>` and :func:`Petrosian fractal dimension <fractal_petrosian>`, different methods to transform continuous signals into discrete ones are available. See :func:`fractal_petrosian` for details.

    This function can be called either via ``entropy_cumulative_residual()`` or ``complexity_cren()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str or int
        Method of discretization. Can be one of ``"A"``, ``"B"``, ``"C"``, ``"D"``, ``"r"``, an
        ``int`` indicating the number of bins, or ``None`` to skip the process (for instance, in
        cases when the binarization has already been done before). See :func:`fractal_petrosian`
        for details.
    show : bool
        If ``True``, will show the discrete the signal.

    Returns
    -------
    CREn : float
        The cumulative residual entropy.
    info : dict
        A dictionary containing 'Values' for each pair of events.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = [1, 1, 1, 3, 3, 2, 2]
      @savefig p_entropy_cumulative1.png scale=100%
      cren, info = nk.entropy_cumulative_residual(signal, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      cren

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Check if string ('ABBA'), and convert each character to list (['A', 'B', 'B', 'A'])
    if isinstance(signal, str):
        signal = list(signal)

    # Force to array
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    # Make discrete
    if np.isscalar(signal) is False:
        signal, _ = _complexity_binarize(signal, method=method, show=show)

    # Get probability of each event
    valscount = pd.Series(signal).value_counts(sort=True)
    events, probs = valscount.index.values, valscount.values / valscount.sum()

    cdf = {a: _ for a, _ in zip(events, np.cumsum(probs))}
    terms = np.zeros(len(events))
    for i, (a, b) in enumerate(_entropy_cumulative_residual_pairwise(events)):
        pgx = cdf[a]
        term = (b - a) * pgx * np.log2(pgx)
        terms[i] = term
    return -np.nansum(terms), {"Values": terms, "Method": method}


# =============================================================================
# Utilities
# =============================================================================
def _entropy_cumulative_residual_pairwise(events):
    pairs = itertools.tee(events, 2)
    pairs = list(zip(*pairs))
    for i, _ in enumerate(pairs[:-1]):
        pairs[i] = (pairs[i][0], pairs[i + 1][0])
    return pairs[:-1]
