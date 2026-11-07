import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial

from ..misc import expspace
from .utils_complexity_embedding import complexity_embedding


def fractal_correlation(signal, delay=1, dimension=2, radius=64, theiler=0, show=False, **kwargs):
    """**Correlation Dimension (CD)**

    The Correlation Dimension (CD, also denoted *D2*) is a lower bound estimate of the fractal
    dimension of a signal.

    The time series is first :func:`time-delay embedded <complexity_embedding>`, and distances
    between all points in the trajectory are calculated. The "correlation sum" is then computed,
    which is the proportion of pairs of points whose distance is smaller than a given radius. The
    final correlation dimension is then approximated by a log-log graph of correlation sum vs. a
    sequence of radiuses (Grassberger & Procaccia, 1983).

    In oversampled signals (e.g., EEG), consecutive points of the trajectory are close because of
    the autocorrelation of the signal rather than of the geometry of the attractor, which biases
    the correlation sum at small radii. The usual remedy is to leave out the pairs of points closer
    than *w* samples in time, the Theiler window (Theiler, 1986), via the ``theiler`` argument
    (e.g., the delay, or the autocorrelation time of the signal).

    Note that the default radii (2.5% to 50% of the largest distance) can reach beyond the scaling
    region of many attractors, where the correlation sum saturates, which lowers the estimate
    (e.g., about 1.6 for the Lorenz attractor, against 2.05 in the literature). Radii restricted
    to the scaling region (e.g., 1% to 10% of the largest distance) give a value closer to it.

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
    theiler : int
        Theiler window (*w*): pairs of points closer than ``theiler`` samples in time are left out
        of the correlation sum. Defaults to 0 (all pairs are counted) for consistency with other
        implementations, but a non-zero value is recommended for oversampled signals (e.g., EEG),
        where it removes a bias that can lower the estimate substantially. Sensible choices are the
        ``delay`` (an estimate of the decorrelation time, see :func:`complexity_delay`) or, more
        strictly, the embedding window ``delay * (dimension - 1)``, beyond which the points no
        longer share samples. With ``delay=1``, a window of 1 changes the result negligibly.
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
    * Grassberger, P., & Procaccia, I. (1983). Measuring the strangeness of strange attractors.
      Physica D: Nonlinear Phenomena, 9(1-2), 189-208.
    * Theiler, J. (1986). Spurious dimension from correlation algorithms applied to limited
      time-series data. Physical Review A, 34(3), 2427-2432.
    * Bolea, J., Laguna, P., Remartínez, J. M., Rovira, E., Navarro, A., & Bailón, R. (2014).
      Methodological framework for estimating the correlation dimension in HRV signals.
      Computational and mathematical methods in medicine, 2014.
    * Boon, M. Y., Henry, B. I., Suttle, C. M., & Dain, S. J. (2008). The correlation dimension:
      A useful objective measure of the transient visual evoked potential?. Journal of vision,
      8(1), 6-6.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    # Get embedded
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)

    # Distances between all pairs of points, each pair once (condensed form)
    dist = scipy.spatial.distance.pdist(embedded)

    # A constant signal has no distances, so its dimension is 0
    if len(dist) == 0 or np.max(dist) == 0:
        return 0.0, {"Dimension": dimension, "Delay": delay, "Radius": np.array([])}

    # The radii are based on all the distances, so that they do not depend on the Theiler window
    r_vals = np.sort(_fractal_correlation_get_r(radius, signal, dist))

    # Store parameters
    info = {"Dimension": dimension, "Delay": delay, "Radius": r_vals}

    if theiler > 0:
        # Leave out the pairs closer than `theiler` samples in time (Theiler, 1986): they are close
        # because of the autocorrelation of the signal, not of the geometry of the attractor
        i, j = np.triu_indices(len(embedded), 1)
        dist = dist[(j - i) > theiler]
        if len(dist) == 0:
            return np.nan, info

    # Correlation sums of all radii in one pass: the proportion of distances below each radius
    counts = np.bincount(np.searchsorted(r_vals, dist, side="right"), minlength=len(r_vals) + 1)
    corr = np.cumsum(counts)[: len(r_vals)] / len(dist)

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
    if isinstance(radius, (list, np.ndarray)):
        return np.asarray(radius, dtype=float)

    if isinstance(radius, str):
        if radius == "nolds":
            sd = np.std(signal, ddof=1)
            min_r, max_r, factor = 0.1 * sd, 0.5 * sd, 1.03

            r_n = int(np.floor(np.log(1.0 * max_r / min_r) / np.log(factor)))
            r_vals = np.array([min_r * (factor**i) for i in range(r_n + 1)])

        elif radius == "Corr_Dim":
            r_min, r_max = np.min(dist[np.where(dist > 0)]), np.exp(np.floor(np.log(np.max(dist))))

            n_r = int(np.floor(np.log(r_max / r_min))) + 1

            ones = -1 * np.ones([n_r])
            r_vals = r_max * np.exp(ones * np.arange(n_r) - ones)

        elif radius == "boon2008":
            r_min, r_max = np.min(dist[np.where(dist > 0)]), np.max(dist)
            r_vals = r_min + np.arange(1, 65) * ((r_max - r_min) / 64)

    if isinstance(radius, int):
        # From 2.5% to 50% of the largest distance
        r_min, r_max = 0.025 * np.max(dist), 0.5 * np.max(dist)
        r_vals = expspace(r_min, r_max, radius, base=2, out="float")

    return r_vals
