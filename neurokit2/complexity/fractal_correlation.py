# -*- coding: utf-8 -*-
import numpy as np
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt

from .embedding import embedding


def fractal_correlation(signal, method="nolds", delay=1, dimension=2, show=False):
    """Correlation Dimension

    Parameters
    ----------
    signal : list, array or Series
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically 2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns an array with two columns corresponding to the original signal and its delayed (by Tau) version.

    Returns
    ----------
    D : float
        The correlation dimension.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>>
    >>> nk.fractal_correlation(signal, method="nolds", show=True)
    >>> nk.fractal_correlation(signal, method="Corr_Dim", show=True)


    References
    -----------
    - `nolds <https://github.com/CSchoel/nolds/blob/master/nolds/measures.py>`_
    """
    embedded = embedding(signal, delay=delay, dimension=dimension)
    dist = sklearn.metrics.pairwise.euclidean_distances(embedded)

    if method == "nolds":
        r_vals = _fractal_correlation_get_r(signal, method=1)
        r_vals, corr = _fractal_correlation1(signal, r_vals, dist)
    else:
        r_vals = _fractal_correlation_get_r(dist, method=2)
        r_vals, corr = _fractal_correlation2(embedded, r_vals, dist)


    # Compute trend
    if len(corr) == 0:
        return np.nan
    else:
        d = np.polyfit(np.log2(r_vals), np.log2(corr), 1)


    if show is True:
        _fractal_correlation_plot(r_vals, corr, d)

    return d[0]


# =============================================================================
# Methods
# =============================================================================
def _fractal_correlation1(signal, r_vals, dist):
    """
    References
    -----------
    - `nolds <https://github.com/CSchoel/nolds/blob/master/nolds/measures.py>`_
    """
    n = len(signal)


    corr = np.zeros(len(r_vals))
    for i, r in enumerate(r_vals):
        corr[i] = 1 / (n * (n - 1)) * np.sum(dist < r)


    # filter zeros from csums
    nonzero = np.nonzero(corr)[0]
    r_vals = r_vals[nonzero]
    corr = corr[nonzero]

    return r_vals, corr



def _fractal_correlation2(embedded, r_vals, dist):
    """
    References
    -----------
    - `Corr_Dim <https://github.com/jcvasquezc/Corr_Dim>`_
    """
    ED = dist[np.triu_indices_from(dist, k=1)]

    Npairs = ((len(embedded[1, :])) * ((len(embedded[1, :]) - 1)))
    corr = np.zeros(len(r_vals))

    for i, r in enumerate(r_vals):
        N = np.where(((ED < r) & (ED>0)))
        corr[i] = len(N[0]) / Npairs

    omit_pts = 1
    k1 = omit_pts
    k2 = len(r_vals) - omit_pts
    r_vals = r_vals[k1:k2]
    corr = corr[k1:k2]

    return r_vals, corr


# =============================================================================
# Utilities
# =============================================================================
def _fractal_correlation_get_r(signal, method=1):
    if method == 1:
        sd = np.std(signal, ddof=1)
        r_vals = _fractal_correlation_range_r(0.1 * sd, 0.5 * sd, 1.03)
    else:
        max_r = np.max(signal)
        min_r = np.min(signal[np.where(signal > 0)])
        max_r = np.exp(np.floor(np.log(max_r)))
        n_r = np.int(np.floor(np.log(max_r / min_r))) + 1

        ones = -1 * np.ones([n_r])
        r_vals = max_r * np.exp(ones * np.arange(n_r) - ones)

    return r_vals






def _fractal_correlation_range_r(min_n, max_n, factor):
  """
  Creates a list of values by successively multiplying a minimum value min_n by
  a factor > 1 until a maximum value max_n is reached.
  Args:
    min_n (float):
      minimum value (must be < max_n)
    max_n (float):
      maximum value (must be > min_n)
    factor (float):
      factor used to increase min_n (must be > 1)
  Returns:
    list of floats:
      min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
  """
  assert max_n > min_n
  assert factor > 1

  max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))

  return np.array([min_n * (factor ** i) for i in range(max_i + 1)])





def _fractal_correlation_plot(r_vals, corr, d):
    fit = 2**np.polyval(d, np.log2(r_vals))
    plt.loglog(r_vals, corr, 'bo')
    plt.loglog(r_vals, fit, 'r', label=r'$D$ = %0.3f' % d[0])
    plt.title('Correlation Dimension')
    plt.xlabel(r'$\log_{10}$(r)')
    plt.ylabel(r'$\log_{10}$(c)')
    plt.legend()
    plt.show()
