import numpy as np
import pandas as pd
import scipy.special

from .entropy_shannon import _entropy_freq


def entropy_cumulativeresidual(signal=None, symbolize=None, show=False, freq=None, **kwargs):
    """**Cumulative residual entropy (CREn)**

    The cumulative residual entropy is an alternative to the Shannon
    differential entropy with several advantageous properties, such as non-negativity. The key idea
    is to use the cumulative distribution (CDF) instead of the density function in Shannon's
    entropy: CREn is computed from the survival function :math:`P(X > x) = 1 - CDF(x)`, i.e., the
    probability that the signal exceeds each value (Rao et al., 2004).

    .. math::

      CREn = -\\int_{0}^{\\infty} P(|X| > x) \\log_{2} P(|X| > x) dx

    It is computed on the empirical distribution of the values of the signal (or of its symbols),
    whose survival function is constant between two successive values *a* and *b*, so that CREn is
    the sum of the terms :math:`(b - a) P(X > a) \\log_{2} P(X > a)` (with a minus sign). For
    non-negative signals (e.g., symbols), this is the integral above. For signals with negative
    values, the survival function of the signal itself is used rather than that of its absolute
    value, so that CREn does not depend on the origin of the values (adding a constant to the
    signal does not change it). CREn is in the units of the signal (multiplying the signal by a
    positive *a* multiplies CREn by *a*): it is, for instance, 1.4427 bits (1 nat) for an
    exponential distribution of mean 1, and 0.3607 bits (1/4 nat) for a uniform distribution
    between 0 and 1.

    Similarly to :func:`Shannon entropy <entropy_shannon>` and :func:`Petrosian fractal dimension
    <fractal_petrosian>`, different methods to transform continuous signals into discrete ones are
    available. See :func:`complexity_symbolize` for details.

    This function can be called either via ``entropy_cumulativeresidual()`` or ``complexity_cren()``.


    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of numeric values.
    symbolize : str
        Method to convert a continuous signal input into a symbolic (discrete) signal. ``None`` by
        default, which skips the process (and uses the values of the signal). See
        :func:`complexity_symbolize` for details.
    show : bool
        If ``True``, will show the discrete signal.
    freq : np.array
        Instead of a signal, a vector of probabilities (or counts, which are normalized) can be
        provided. They are taken as those of the successive values 0, 1, 2, etc. (e.g., the symbols
        returned by :func:`complexity_symbolize`).
    **kwargs
        Other arguments passed to :func:`complexity_symbolize`, such as ``c`` (the number of
        symbols for the methods that need it) or ``random_state``.

    Returns
    -------
    CREn : float
        The cumulative residual entropy. It is NaN if the signal contains missing values.
    info : dict
        A dictionary containing the terms :math:`(b - a) P(X > a) \\log_{2} P(X > a)` of each pair
        of successive values (``"Values"``), of which CREn is minus the sum.

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
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    info = {"Symbolization": symbolize}

    if freq is None:
        events, freq = _entropy_freq(signal, symbolize=symbolize, show=show, **kwargs)
        try:
            events = events.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "CREn requires numeric values (the distances between the values are used). Use the"
                " `symbolize` argument to convert non-numeric states into numeric symbols."
            ) from error
    else:
        freq = np.asarray(freq, dtype=float)
        events = np.arange(len(freq), dtype=float)

    if np.any(np.isnan(events)):
        info["Values"] = np.full(max(len(events) - 1, 0), np.nan)
        return np.nan, info

    freq = freq / np.sum(freq)

    # Survival function P(X > a) between each value a and the next one b (the values are sorted).
    # It is summed from the end so that it is exactly 0 after the last value that occurs.
    survival = np.cumsum(freq[::-1])[::-1][1:]
    terms = np.diff(events) * scipy.special.xlogy(survival, survival) / np.log(2)
    info["Values"] = terms

    return -np.sum(terms), info
