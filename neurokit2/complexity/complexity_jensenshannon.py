import math
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from .utils_complexity_ordinalpatterns import complexity_ordinalpatterns


def complexity_jensenshannon(signal, delay=1, dimension=4, show=False):
    """**Jensen-Shannon Statistical Complexity (C_JS) and the Complexity-Entropy Plane**

    The Jensen-Shannon statistical complexity (Lamberti et al., 2004; Rosso et al., 2007), also
    known as the MPR complexity (after Martin, Plastino & Rosso), is the product of the normalized
    :func:`permutation entropy <entropy_permutation>` *H* of the signal and of the *disequilibrium*
    *Q*, the (normalized) Jensen-Shannon divergence between the distribution of its ordinal
    patterns and the uniform distribution:

    .. math::

      C_{JS} = Q_J[P, P_e] \\cdot H[P], \\quad H[P] = \\frac{S[P]}{\\ln N}, \\quad
      Q_J[P, P_e] = \\frac{S[(P + P_e) / 2] - S[P] / 2 - S[P_e] / 2}{J_{max}}

    where *P* is the distribution of the ordinal patterns, :math:`P_e` the uniform distribution
    over the :math:`N = dimension!` possible patterns, *S* the Shannon entropy and :math:`J_{max}`
    the maximum of the Jensen-Shannon divergence (reached when a single pattern occurs), so that
    *Q* and :math:`C_{JS}` lie between 0 and 1.

    The complexity is 0 both for perfectly ordered signals (a single pattern, *H* = 0) and for
    fully random ones (all patterns equally likely, *Q* = 0), and is highest for structured signals
    in between. Unlike *H* alone, it distinguishes chaotic signals from (coloured) noise of the
    same entropy: in the **complexity-entropy plane** (*H*, :math:`C_{JS}`), chaotic maps lie close
    to the maximum complexity attainable at their entropy, while stochastic processes lie lower
    (Rosso et al., 2007). The plane is bounded by the minimum and maximum complexities possible
    for a given entropy (Martin et al., 2006), drawn with ``show=True``.

    The proportion of the possible patterns that never occur (*missing* or *forbidden* patterns)
    is also returned. Deterministic dynamics forbid some patterns, whereas all of them eventually
    appear in a random signal (Amigó et al., 2007), as long as it is long enough: the signal should
    be much longer than the number of possible patterns, :math:`dimension!` (e.g., 24 for the
    default ``dimension=4``, 120 for 5 and 720 for 6). Rosso et al. (2007) recommend a dimension
    between 3 and 7; with a dimension of 3, *C* is nearly a function of *H*.

    This is not the statistical complexity of computational mechanics (Crutchfield & Young, 1989),
    the entropy of the causal states of a process.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*), i.e., the length of the
        ordinal patterns. See :func:`complexity_dimension` to estimate the optimal value for this
        parameter.
    show : bool
        If ``True``, plots the signal in the complexity-entropy plane, with the curves of minimum
        and maximum complexity.

    Returns
    -------
    C : float
        The Jensen-Shannon statistical complexity, between 0 and 1.
    info : dict
        A dictionary containing additional information, such as the normalized permutation entropy
        (``"Entropy"``, *H*), the disequilibrium (``"Disequilibrium"``, *Q*), and the proportion of
        possible patterns that never occur (``"MissingPatterns"``).

    See Also
    --------
    entropy_permutation, complexity_ordinalpatterns, fishershannon_information

    Examples
    ----------
    .. ipython:: python

      import numpy as np
      import neurokit2 as nk

      # Chaotic signal: the logistic map with r = 4
      x = np.zeros(5000)
      x[0] = 0.1
      for i in range(1, len(x)):
          x[i] = 4 * x[i - 1] * (1 - x[i - 1])

      @savefig p_complexity_jensenshannon1.png scale=100%
      c, info = nk.complexity_jensenshannon(x, dimension=5, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      c
      info

    White noise has a higher entropy and a near-zero complexity.

    .. ipython:: python

      c, info = nk.complexity_jensenshannon(np.random.normal(size=5000), dimension=5)
      c
      info["Entropy"]

    References
    ----------
    * Lamberti, P. W., Martin, M. T., Plastino, A., & Rosso, O. A. (2004). Intensive entropic
      non-triviality measure. Physica A, 334(1-2), 119-131.
    * Martin, M. T., Plastino, A., & Rosso, O. A. (2006). Generalized statistical complexity
      measures: Geometrical and analytical properties. Physica A, 369(2), 439-462.
    * Rosso, O. A., Larrondo, H. A., Martin, M. T., Plastino, A., & Fuentes, M. A. (2007).
      Distinguishing noise from chaos. Physical Review Letters, 99(15), 154102.
    * Amigó, J. M., Zambrano, S., & Sanjuán, M. A. (2007). True and false forbidden patterns in
      deterministic and random dynamics. Europhysics Letters, 79(5), 50001.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    if not isinstance(dimension, (int, np.integer)) or dimension < 2:
        raise ValueError("NeuroKit error: complexity_jensenshannon(): `dimension` must be an integer of at least 2.")

    info = {"Dimension": dimension, "Delay": delay}
    signal = np.asarray(signal, dtype=float)
    if np.any(np.isnan(signal)):
        warn("The signal contains missing values. Returning NaN.", category=NeuroKitWarning)
        return np.nan, {**info, "Entropy": np.nan, "Disequilibrium": np.nan, "MissingPatterns": np.nan}

    # Distribution of the ordinal patterns
    _, patterns_info = complexity_ordinalpatterns(signal, delay=delay, dimension=dimension)
    p = patterns_info["Frequencies"]
    n = math.factorial(dimension)

    # Observed patterns, and all the missing ones (probability 0)
    h, q = _complexity_jensenshannon(p[np.newaxis, :], np.ones((1, len(p))), n)
    h, q = h[0], q[0]
    c = h * q
    info.update({"Entropy": h, "Disequilibrium": q, "MissingPatterns": 1 - len(p) / n})

    if show is True:
        _complexity_jensenshannon_plot(h, c, n, dimension)

    return c, info


# =============================================================================
# Utilities
# =============================================================================
def _complexity_jensenshannon(probabilities, counts, n):
    """Normalized entropy and Jensen-Shannon complexity of distributions over n states.

    Each distribution (row) is described by groups of states sharing the same probability
    (``probabilities``) and their number (``counts``). The states not counted have a probability
    of 0. Returns the normalized Shannon entropy H and the disequilibrium Q (C = H * Q).
    """
    probabilities = np.asarray(probabilities, dtype=float)
    counts = np.asarray(counts, dtype=float)
    missing = n - counts.sum(axis=1)

    def _plogp(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x > 0, x * np.log(x), 0.0)

    s = 0.0 - np.sum(counts * _plogp(probabilities), axis=1)  # (avoids -0.0)
    # Mixture of each distribution with the uniform one
    s_mixture = -np.sum(counts * _plogp((probabilities + 1 / n) / 2), axis=1) - missing * _plogp(1 / (2 * n))
    js = s_mixture - s / 2 - np.log(n) / 2
    # Maximum of the divergence (a single state of probability 1)
    js_max = -0.5 * (((n + 1) / n) * np.log(n + 1) + np.log(n) - 2 * np.log(2 * n))

    h = s / np.log(n)
    q = np.clip(js / js_max, 0, 1)  # Rounding errors
    return h, q


def _complexity_jensenshannon_curves(n, n_points=100):
    """Minimum and maximum complexity as a function of the normalized entropy, for n states
    (Martin et al., 2006)."""
    # Minimum: one state of probability p (from 1/n to 1), the others sharing 1 - p
    p = np.linspace(1 / n, 1, n_points)
    h_min, q_min = _complexity_jensenshannon(np.c_[p, (1 - p) / (n - 1)], np.tile([1, n - 1], (n_points, 1)), n)

    # Maximum: k states with a non-zero probability (k from 2 to n), one of which has a probability
    # p (from 0 to 1/k) and the k - 1 others share 1 - p (the other n - k states are empty)
    k = np.repeat(np.arange(2, n + 1), n_points)
    p = np.concatenate([np.linspace(0, 1 / kk, n_points) for kk in range(2, n + 1)])
    counts = np.c_[np.ones(len(k)), k - 1]
    h_max, q_max = _complexity_jensenshannon(np.c_[p, (1 - p) / (k - 1)], counts, n)
    order = np.argsort(h_max)

    return (h_min, h_min * q_min), (h_max[order], (h_max * q_max)[order])


def _complexity_jensenshannon_plot(h, c, n, dimension):
    (h_min, c_min), (h_max, c_max) = _complexity_jensenshannon_curves(n)
    plt.plot(h_min, c_min, color="#9E9E9E", linestyle="--", label="Minimum complexity")
    plt.plot(h_max, c_max, color="#9E9E9E", label="Maximum complexity")
    plt.scatter([h], [c], color="#E91E63", zorder=3, label=f"Signal (H = {h:.3f}, C = {c:.3f})")
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    plt.xlabel("Normalized Permutation Entropy (H)")
    plt.ylabel("Statistical Complexity (C)")
    plt.title(f"Complexity-Entropy Plane (dimension = {dimension})")
    plt.legend()
