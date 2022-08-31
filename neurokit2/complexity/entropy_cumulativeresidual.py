import itertools

import numpy as np
import pandas as pd

from .entropy_shannon import _entropy_freq


def entropy_cumulativeresidual(signal, symbolize=None, show=False, freq=None):
    """**Cumulative residual entropy (CREn)**

    The cumulative residual entropy is an alternative to the Shannon
    differential entropy with several advantageous properties, such as non-negativity. The key idea
    is to use the cumulative distribution (CDF) instead of the density function in Shannon's
    entropy.

    .. math::

      CREn = -\\int_{0}^{\\infty} p(|X| > x) \\log_{2} p(|X| > x) dx

    Similarly to :func:`Shannon entropy <entropy_shannon>` and :func:`Petrosian fractal dimension
    <fractal_petrosian>`, different methods to transform continuous signals into discrete ones are
    available. See :func:`complexity_symbolize` for details.

    This function can be called either via ``entropy_cumulativeresidual()`` or ``complexity_cren()``.


    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    symbolize : str
        Method to convert a continuous signal input into a symbolic (discrete) signal. ``None`` by
        default, which skips the process (and assumes the input is already discrete). See
        :func:`complexity_symbolize` for details.
    show : bool
        If ``True``, will show the discrete the signal.
    freq : np.array
        Instead of a signal, a vector of probabilities can be provided.

    Returns
    -------
    CREn : float
        The cumulative residual entropy.
    info : dict
        A dictionary containing ``Values`` for each pair of events.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = [1, 1, 1, 3, 3, 2, 2, 1, 1, 3, 3, 3]

      @savefig p_entropy_cumulativeresidual1.png scale=100%
      cren, info = nk.entropy_cumulativeresidual(signal, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      cren

    References
    -----------
    * Rao, M., Chen, Y., Vemuri, B. C., & Wang, F. (2004). Cumulative residual entropy: a new
      measure of information. IEEE transactions on Information Theory, 50(6), 1220-1228.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    if freq is None:
        events, freq = _entropy_freq(signal, symbolize=symbolize, show=show)
    freq = freq / np.sum(freq)

    events, freq = zip(*sorted(zip(events, freq)))

    # Get the CDF
    cdf = {a: _ for a, _ in zip(events, np.cumsum(freq))}
    terms = np.zeros(len(events))
    for i, (a, b) in enumerate(_entropy_cumulativeresidual_pairwise(events)):
        pgx = cdf[a]
        term = (b - a) * pgx * np.log2(pgx)
        terms[i] = term

    return -np.nansum(terms), {"Values": terms, "Symbolization": symbolize}


# =============================================================================
# Utilities
# =============================================================================
def _entropy_cumulativeresidual_pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
