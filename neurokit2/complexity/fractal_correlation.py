# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise

from ..misc import expspace
from .utils_complexity_embedding import complexity_embedding


def fractal_correlation(signal, delay=1, dimension=2, radius=64, show=False, **kwargs):
    """**Correlation Dimension (CD)**

    The Correlation Dimension (CD, also denoted *D2*) is a lower bound estimate of the fractal
    dimension of a signal.

    The time series is first :func:`time-delay embedded <complexity_embedding>`, and distances
    between all points in the trajectory are calculated. The "correlation sum" is then computed,
    which is the proportion of pairs of points whose distance is smaller than a given radius. The
    final correlation dimension is then approximated by a log-log graph of correlation sum vs. a
    sequence of radiuses.

    This function can be called either via ``fractal_correlation()`` or ``complexity_cd()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    radius : Union[str, int, list]
        The sequence of radiuses to test. If an integer is passed, will get an exponential sequence
        of length ``radius`` ranging from 2.5% to 50% of the distance range. Methods implemented in
        other packages can be used via ``"nolds"``, ``"Corr_Dim"`` or ``"boon2008"``.
    show : bool
        Plot of correlation dimension if ``True``. Defaults to ``False``.
    **kwargs
        Other arguments to be passed (not used for now).

    Returns
    ----------
    cd : float
        The Correlation Dimension (CD) of the time series.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute the correlation dimension.

    Examples
    ----------
    For some completely unclear reasons, uncommenting the following examples messes up the figures
    path of all the subsequent documented function. So, commenting it for now.

    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=1, frequency=[10, 14], noise=0.1)

      # @savefig p_fractal_correlation1.png scale=100%
      # cd, info = nk.fractal_correlation(signal, radius=32, show=True)
      # @suppress
      # plt.close()

    .. ipython:: python

      # @savefig p_fractal_correlation2.png scale=100%
      # cd, info = nk.fractal_correlation(signal, radius="nolds", show=True)
      # @suppress
      # plt.close()

    .. ipython:: python

      # @savefig p_fractal_correlation3.png scale=100%
      # cd, info = nk.fractal_correlation(signal, radius='boon2008', show=True)
      # @suppress
      # plt.close()

    References
    -----------
    * Bolea, J., Laguna, P., Remartínez, J. M., Rovira, E., Navarro, A., & Bailón, R. (2014).
      Methodological framework for estimating the correlation dimension in HRV signals.
      Computational and mathematical methods in medicine, 2014.
    * Boon, M. Y., Henry, B. I., Suttle, C. M., & Dain, S. J. (2008). The correlation dimension:
      A useful objective measure of the transient visual evoked potential?. Journal of vision,
      8(1), 6-6.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Get embedded
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    dist = sklearn.metrics.pairwise.euclidean_distances(embedded)
    r_vals = _fractal_correlation_get_r(radius, signal, dist)

    # Store parameters
    info = {"Dimension": dimension, "Delay": delay, "Radius": r_vals}

    # Get only upper triang of the distance matrix to reduce computational load
    upper = dist[np.triu_indices_from(dist, k=1)]
    corr = np.array([np.sum(upper < r) for r in r_vals])
    corr = corr / len(upper)

    # filter zeros from correlation sums
    r_vals = r_vals[np.nonzero(corr)[0]]
    corr = corr[np.nonzero(corr)[0]]

    # Compute trend
    if len(corr) == 0:
        return np.nan, info
    else:
        cd, intercept = np.polyfit(np.log2(r_vals), np.log2(corr), 1)

    if show is True:
        plt.figure()
        plt.title("Correlation Dimension")
        plt.xlabel(r"$\log_{2}$(radius)")
        plt.ylabel(r"$\log_{2}$(correlation sum)")

        fit = 2 ** np.polyval((cd, intercept), np.log2(r_vals))
        plt.loglog(r_vals, corr, "bo")
        plt.loglog(r_vals, fit, "r", label=f"$CD$ = {np.round(cd, 2)}")
        plt.legend(loc="lower right")

    return cd, info


# =============================================================================
# Utilities
# =============================================================================
def _fractal_correlation_get_r(radius, signal, dist):
    if isinstance(radius, str):
        if radius == "nolds":
            sd = np.std(signal, ddof=1)
            min_r, max_r, factor = 0.1 * sd, 0.5 * sd, 1.03

            r_n = int(np.floor(np.log(1.0 * max_r / min_r) / np.log(factor)))
            r_vals = np.array([min_r * (factor ** i) for i in range(r_n + 1)])

        elif radius == "Corr_Dim":
            r_min, r_max = np.min(dist[np.where(dist > 0)]), np.exp(np.floor(np.log(np.max(dist))))

            n_r = int(np.floor(np.log(r_max / r_min))) + 1

            ones = -1 * np.ones([n_r])
            r_vals = r_max * np.exp(ones * np.arange(n_r) - ones)

        elif radius == "boon2008":
            r_min, r_max = np.min(dist[np.where(dist > 0)]), np.max(dist)
            r_vals = r_min + np.arange(1, 65) * ((r_max - r_min) / 64)

    if isinstance(radius, int):
        dist_range = np.max(dist) - np.min(dist)
        r_min, r_max = (np.min(dist) + 0.025 * dist_range), (np.min(dist) + 0.5 * dist_range)
        r_vals = expspace(r_min, r_max, radius, base=2, out="float")

    return r_vals
