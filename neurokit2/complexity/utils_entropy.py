import numpy as np
import sklearn.metrics
import sklearn.neighbors
from packaging import version

from .utils_complexity_embedding import complexity_embedding


# Distance metrics accepted by the KDTree (computed once: in sklearn 1.3.0, valid_metrics was a method)
if version.parse(sklearn.__version__) in [version.parse("1.3.0"), version.parse("1.3.0rc1")]:
    _VALID_METRICS = sklearn.neighbors.KDTree.valid_metrics() + ["range"]
else:
    _VALID_METRICS = sklearn.neighbors.KDTree.valid_metrics + ["range"]

# Up to this number of embedded vectors, counting the neighbours with a dense distance matrix is
# faster than building a KDTree, whose fixed overhead dominates on short series (e.g., the
# coarse-grained series of multiscale entropy)
_DENSE_COUNT_MAX_N = 50


# =============================================================================
# ApEn
# =============================================================================
def _entropy_apen(signal, delay, dimension, tolerance, **kwargs):
    phi, info = _phi(
        signal,
        delay=delay,
        dimension=dimension,
        tolerance=tolerance,
        approximate=True,
        **kwargs,
    )

    # Pincus (1991): ApEn = Phi^m - Phi^(m+1). Keep the sign: the difference is negative when
    # the vectors match only themselves (self-match bias), which is a degenerate estimate
    # that the absolute value would otherwise turn into a small positive value.
    return phi[0] - phi[1], info


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
    kdtree1=None,
    kdtree2=None,
    **kwargs,
):
    """Common internal for `entropy_approximate`, `entropy_sample` and `entropy_range`."""

    # Embed signal at m and m+1
    embedded1, count1, kdtree1 = _get_count(
        signal,
        delay,
        dimension,
        tolerance,
        distance=distance,
        approximate=approximate,
        fuzzy=fuzzy,
        kdtree=kdtree1,
    )

    embedded2, count2, kdtree2 = _get_count(
        signal,
        delay,
        dimension + 1,
        tolerance,
        distance=distance,
        approximate=True,
        fuzzy=fuzzy,
        kdtree=kdtree2,
    )

    # Initialize phi
    phi = np.zeros(2)
    if approximate is True:
        phi[0] = np.mean(np.log(count1 / embedded1.shape[0]))
        phi[1] = np.mean(np.log(count2 / embedded2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (embedded1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (embedded2.shape[0] - 1))

    return phi, {
        "embedded1": embedded1,
        "count1": count1,
        "kdtree1": kdtree1,
        "embedded2": embedded2,
        "count2": count2,
        "kdtree2": kdtree2,
    }


def _phi_divide(phi):
    # Same absolute tolerance as np.isclose(x, 0), which is much slower
    if abs(phi[0]) <= 1e-8:
        return -np.inf
    division = phi[1] / phi[0]
    if abs(division) <= 1e-8:
        return np.inf
    if division < 0:
        return np.nan
    return -np.log(division)


# =============================================================================
# Get Embedded
# =============================================================================


def _get_count(
    signal,
    delay=1,
    dimension=2,
    tolerance=0.1,
    distance="chebyshev",
    approximate=True,
    fuzzy=False,
    kdtree=None,
    n=1,
    **kwargs,
):
    """
    This is usually the bottleneck for several complexity methods, in particular in the counting.
    That's why we allow the possibility of giving kdtrees as pre-computed (used in the optimization
    of tolerance via MaxApEn which computes iteratively the value with multiple tolerances).
    However, more improvements are welcome!
    """
    # Get embedded
    # -------------------
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    if approximate is False:
        # SampEn (Richman & Moorman, 2000) uses the same templates at m and m + 1: keep the first
        # N - m * delay of the N - (m - 1) * delay vectors, i.e., those that also exist at m + 1
        embedded = embedded[:-delay]

    # Get neighbors count
    # -------------------
    # Sanity checks
    if distance not in _VALID_METRICS:
        raise ValueError(f"The given metric ({distance}) is not valid. Valid metric names are: {_VALID_METRICS}")

    if fuzzy is True:
        if distance == "range":
            raise ValueError("The fuzzy option is not available for range distance.")

        # FuzzyEn: Remove the local baselines of vectors
        embedded -= np.mean(embedded, axis=1, keepdims=True)

        # TODO: it would be good to implement 'distrange' here to have fuzzy RangeEn
        # TODO: also, different membership functions?
        # https://github.com/HamedAzami/FuzzyEntropy_Matlab/blob/master/FuzEn_MFs.m
        dist = sklearn.metrics.DistanceMetric.get_metric(distance)
        dist = dist.pairwise(embedded)
        # sklearn.metrics.pairwise_distances_chunked()
        if n > 1:
            sim = np.exp(-(dist**n) / tolerance)
        else:
            sim = np.exp(-dist / tolerance)
        # Return the count
        count = np.sum(sim, axis=0)

    elif distance == "range":
        # internal function for distrange
        def distrange(x, y):
            numerator = np.max(np.abs(x - y), axis=1) - np.min(np.abs(x - y), axis=1)
            denominator = np.max(np.abs(x - y), axis=1) + np.min(np.abs(x - y), axis=1)
            valid = np.where(denominator != 0)  # To prevent division by 0
            return np.divide(numerator[valid], denominator[valid])

        # Count for each row
        count = np.array([np.sum(distrange(embedded, embedded[i]) < tolerance) for i in range(len(embedded))])

    elif kdtree is None and distance == "chebyshev" and len(embedded) <= _DENSE_COUNT_MAX_N:
        # Short series: the dense distance matrix is faster than a KDTree (same counts)
        dist = np.max(np.abs(embedded[:, None, :] - embedded[None, :, :]), axis=2)
        count = np.sum(dist <= tolerance, axis=1).astype(np.float64)

    else:  # chebyshev and other sklearn methods
        # Perhaps scipy.spatial.KDTree would be faster? Especially since its query() method
        # has a `workers` argument to use multiple cores? Benchmark or opinion required!
        if kdtree is None:
            kdtree = sklearn.neighbors.KDTree(embedded, metric=distance)
        count = kdtree.query_radius(embedded, tolerance, count_only=True).astype(np.float64)
    return embedded, count, kdtree
