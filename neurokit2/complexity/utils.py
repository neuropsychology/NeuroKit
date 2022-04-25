# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import sklearn.neighbors

from .complexity_embedding import complexity_embedding

# =============================================================================
# Phi
# =============================================================================


def _phi(
    signal,
    delay=1,
    dimension=2,
    tolerance=0.1,
    distance="chebyshev",
    approximate=True,
    fuzzy=False,
    **kwargs,
):
    """Common internal for `entropy_approximate`, `entropy_sample` and `entropy_range`."""
    # Initialize phi
    phi = np.zeros(2)

    embedded1, count1 = _get_embedded(
        signal, delay, dimension, tolerance, distance=distance, approximate=approximate, fuzzy=fuzzy
    )

    embedded2, count2 = _get_embedded(
        signal, delay, dimension + 1, tolerance, distance=distance, approximate=True, fuzzy=fuzzy
    )

    if approximate is True:
        phi[0] = np.mean(np.log(count1 / embedded1.shape[0]))
        phi[1] = np.mean(np.log(count2 / embedded2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (embedded1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (embedded2.shape[0] - 1))

    return phi


def _phi_divide(phi):
    if phi[0] == 0:
        return -np.inf
    division = np.divide(phi[1], phi[0])
    if division == 0:
        return np.inf
    return -np.log(division)


# =============================================================================
# Get Embedded
# =============================================================================


def _get_embedded(
    signal,
    delay=1,
    dimension=2,
    tolerance=0.1,
    distance="chebyshev",
    approximate=True,
    fuzzy=False,
):
    """Examples
    -----------
    .. ipython:: python

      import neurokit2 as nk
      import sklearn
      complexity_embedding = nk.complexity_embedding

      signal = nk.signal_simulate(duration=2, frequency=5)

      embeded, count = _get_embedded(signal, delay=8, dimension=2, tolerance=0.07,
                                      distance='chebyshev', approximate=False)
    """
    # Sanity checks
    if distance not in sklearn.neighbors.KDTree.valid_metrics + ["range"]:
        raise ValueError(
            "NeuroKit error: _get_embedded(): The given metric (%s) is not valid."
            "The valid metric names are: %s"
            % (distance, sklearn.neighbors.KDTree.valid_metrics + ["range"])
        )

    # Get embedded
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    if approximate is False:
        embedded = embedded[:-1]  # Removes the last line

    if fuzzy is False:
        # Get neighbors count
        count = _get_count(embedded, tolerance=tolerance, distance=distance)
    else:
        # FuzzyEn: Remove the local baselines of vectors
        embedded -= np.mean(embedded, axis=1, keepdims=True)
        count = _get_count_fuzzy(embedded, tolerance=tolerance, distance=distance, n=1)

    return embedded, count


# =============================================================================
# Get Count
# =============================================================================
def _get_count(embedded, tolerance, distance="chebyshev"):
    """Examples
    -----------
    .. ipython:: python

      import neurokit2 as nk
      signal = nk.signal_simulate(duration=1, frequency=[5, 6])
      embedded = nk.complexity_embedding(signal, delay=8, dimension=3)
      tolerance = 0.07
      distance = "range"
      x = embedded
      y = embedded[0]

    """

    if distance == "range":
        # internal function for distrange
        def distrange(x, y):
            numerator = np.max(np.abs(x - y), axis=1) - np.min(np.abs(x - y), axis=1)
            denominator = np.max(np.abs(x - y), axis=1) + np.min(np.abs(x - y), axis=1)
            valid = np.where(denominator != 0)  # To prevent division by 0
            return np.divide(numerator[valid], denominator[valid])

        # Count for each row
        count = np.array(
            [np.sum(distrange(embedded, embedded[i]) < tolerance) for i in range(len(embedded))]
        )

    else:  # chebyshev and other sklearn methods
        # Perhaps scipy.spatial.KDTree would be faster? Especially since its query() method
        # has a `workers` argument to use multiple cores? Benchmark or opinion required!
        kdtree = sklearn.neighbors.KDTree(embedded, metric=distance)
        count = kdtree.query_radius(embedded, tolerance, count_only=True).astype(np.float64)

    return count


def _get_count_fuzzy(embedded, tolerance, distance="chebyshev", n=1):
    # TODO: it would be good to implement 'distrange' here to have fuzzy RangeEn
    dist = sklearn.metrics.DistanceMetric.get_metric(distance)
    dist = dist.pairwise(embedded)

    if n > 1:
        sim = np.exp(-(dist ** n) / tolerance)
    else:
        sim = np.exp(-dist / tolerance)
    # Return the count
    return np.sum(sim, axis=0)


# =============================================================================
# Get R
# =============================================================================
def _get_tolerance(signal, tolerance="default", dimension=2, show=False):
    """Sanitize the tolerance r For the default value, following the suggestion by Christopher
    Schölzel (nolds), we make it to take into account the number of dimensions. Additionally, a
    constant is introduced, so that for dimension=2, tolerance = 0.2 * np.std(signal, ddof=1),
    which is the traditional default value.

    See nolds for more info:
    https://github.com/CSchoel/nolds/blob/d8fb46c611a8d44bdcf21b6c83bc7e64238051a4/nolds/measures.py#L752

    """

    def _default_tolerance(signal, dimension):
        constant = 0.11604738531196232
        r = constant * np.std(signal, ddof=1) * (0.5627 * np.log(dimension) + 1.3334)
        return r

    # r = "default"
    if isinstance(tolerance, str) or (tolerance is None):
        # Get different r values per channel and find mean
        if signal.ndim > 1:
            r_list = []
            for i, col in enumerate(signal):
                value = _default_tolerance(signal[col], dimension=dimension)
                r_list.append(value)
            optimal_r = np.mean(r_list)

            if show:
                fig = plt.figure(constrained_layout=False)
                fig.suptitle("Sanitized tolerance r across channels")
                colors = plt.cm.plasma(np.linspace(0, 1, len(r_list)))
                plt.plot(np.arange(1, len(r_list) + 1), np.array(r_list), color="#FF9800")
                plt.ylabel(r"Tolerance $r$")
                plt.xlabel("Channels")
                plt.xticks(np.arange(1, len(r_list) + 1), labels=list(signal.columns))
                for i, val in enumerate(r_list):
                    plt.scatter(
                        i + 1, val, color=colors[i], marker="o", zorder=3, label=signal.columns[i]
                    )
                    plt.legend(loc="lower right")
                plt.axhline(optimal_r, color="black", ls="--")
                plt.text(
                    len(r_list) - 1,
                    optimal_r,
                    r"Mean $r$ = {:.3g}".format(optimal_r),
                    ha="center",
                    va="bottom",
                )

        else:
            # one r for single time series
            optimal_r = _default_tolerance(signal, dimension=dimension)
    else:
        optimal_r = tolerance

    return optimal_r
