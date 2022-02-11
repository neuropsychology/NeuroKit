# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise

from ..misc import expspace
from .complexity_embedding import complexity_embedding


def fractal_correlation(signal, delay=1, dimension=2, radius=64, show=False, **kwargs):
    """Correlation Dimension.

    The time series is first reconstructed using a delay-embedding method. In the reconstructed
    phase space trajectory, distances between all points in the trajectory are calculated. The
    'correlation sum' is the computed, which is the probability of finding two vectors which are
    separated by a distance not larger than a specified radius. The final correlation dimension is
    then approximated by a log-log graph of correlation sum vs. a sequence of radiuses.

    Python implementation of the Correlation Dimension CD (sometimes referred to as D2) of a signal.
    This function can be called either via ``fractal_correlation()`` or ``complexity_cd()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common
        to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003),
        or to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding
        returns an array with two columns corresponding to the original signal and its delayed (by
        Tau) version.
    radius : Union[str, int, list]
        The sequence of radiuses to test. If an integer is passed, will get an exponential sequence
        of length ``radius`` ranging from 2.5% to 50% of the distance range. Methods implemented in
        other packages can be used via setting ``r='nolds'``, ``r='Corr_Dim'`` or ``r='boon2008'``.
    show : bool
        Plot of correlation dimension if True. Defaults to False.
    **kwargs
        Other arguments to be passed (unused for now).

    Returns
    ----------
    cd : float
        The correlation dimension of the single time series.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute the correlation dimension.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>>
    >>> fractal1, info = nk.fractal_correlation(signal, radius=32, show=True)
    >>> fractal1 # doctest: +ELLIPSIS
    0.7888...
    >>>
    >>> fractal2, info = nk.fractal_correlation(signal, radius="nolds", show=True)
    >>> fractal2 # doctest: +ELLIPSIS
    0.8316...
    >>>
    >>> fractal3, info = nk.fractal_correlation(signal, radius='boon2008', show=True)
    >>> fractal3 # doctest: +ELLIPSIS
    0.7501...


    References
    -----------
    - Bolea, J., Laguna, P., Remartínez, J. M., Rovira, E., Navarro, A., & Bailón, R. (2014).
      Methodological framework for estimating the correlation dimension in HRV signals. Computational
      and mathematical methods in medicine, 2014.

    - Boon, M. Y., Henry, B. I., Suttle, C. M., & Dain, S. J. (2008). The correlation dimension:
      A useful objective measure of the transient visual evoked potential?. Journal of vision, 8(1), 6-6.

    - `nolds <https://github.com/CSchoel/nolds/blob/master/nolds/measures.py>`_

    - `Corr_Dim <https://github.com/jcvasquezc/Corr_Dim>`_

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Prepare parameters
    info = {"Dimension": dimension, "Delay": delay}

    out, info["Radiuses"] = _fractal_correlation(
        signal, delay=delay, dimension=dimension, radius=radius, show=show
    )

    return out, info


def _fractal_correlation(signal, delay=1, dimension=2, radius=64, show=True):

    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    dist = sklearn.metrics.pairwise.euclidean_distances(embedded)
    r_vals = _fractal_correlation_get_r(radius, signal, dist)

    r_vals, corr = _fractal_correlation_nolds(signal, r_vals, dist)
    # Corr_Dim method: https://github.com/jcvasquezc/Corr_Dim
    # r_vals, corr = _fractal_correlation_Corr_Dim(embedded, r_vals, dist)

    # Compute trend
    if len(corr) == 0:
        return np.nan
    else:
        cd = np.polyfit(np.log2(r_vals), np.log2(corr), 1)

    if show is True:
        _fractal_correlation_plot(r_vals, corr, cd)

    return cd[0], r_vals


# =============================================================================
# Methods
# =============================================================================
def _fractal_correlation_nolds(signal, r_vals, dist):
    """References
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


def _fractal_correlation_Corr_Dim(embedded, r_vals, dist):
    """References
    -----------
    - `Corr_Dim <https://github.com/jcvasquezc/Corr_Dim>`_
    """
    ED = dist[np.triu_indices_from(dist, k=1)]

    Npairs = (len(embedded[1, :])) * ((len(embedded[1, :]) - 1))
    corr = np.zeros(len(r_vals))

    for i, r in enumerate(r_vals):
        N = np.where(((ED < r) & (ED > 0)))
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


def _fractal_correlation_plot(r_vals, corr, d2):

    # Initiate plot
    fig = plt.figure(constrained_layout=False)
    fig.suptitle("Correlation Dimension")
    plt.xlabel(r"$\log_{2}$(r)")
    plt.ylabel(r"$\log_{2}$(c)")

    fit = 2 ** np.polyval(d2, np.log2(r_vals))
    plt.loglog(r_vals, corr, "bo")
    plt.loglog(r_vals, fit, "r", label=r"$D2$ = %0.3f" % d2[0])
    plt.legend(loc="lower right")

    return fig
