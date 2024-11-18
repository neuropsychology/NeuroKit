# -*- coding: utf-8 -*-
import numpy as np
import sklearn.metrics
import sklearn.neighbors

from packaging import version

from .utils_complexity_embedding import complexity_embedding


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

    return np.abs(np.subtract(phi[0], phi[1])), info


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
    func_name="exp",
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
        func_name=func_name,
        block_size=10,
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
        func_name=func_name,
        block_size=10,
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
    if np.isclose(phi[0], 0):
        return -np.inf
    division = np.divide(phi[1], phi[0])
    if np.isclose(division, 0):
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
    func_name="exp",
    kdtree=None,
    **kwargs,
):
    """This is usually the bottleneck for several complexity methods, in particular in the counting.

    That's why we allow the possibility of giving kdtrees as pre-computed (used in the optimization of tolerance
    via MaxApEn which computes iteratively the value with multiple tolerances). However, more improvements are
    welcome!

    """
    # Get embedded
    # -------------------
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    if approximate is False:
        embedded = embedded[:-1]  # Removes the last line

    # Get neighbors count
    # -------------------
    # Sanity checks
    sklearn_version = version.parse(sklearn.__version__)
    if sklearn_version in [version.parse("1.3.0"), version.parse("1.3.0rc1")]:
        valid_metrics = sklearn.neighbors.KDTree.valid_metrics() + ["range"]
    else:
        valid_metrics = sklearn.neighbors.KDTree.valid_metrics + ["range"]
    if distance not in valid_metrics:
        raise ValueError(f"The given metric ({distance}) is not valid." f" Valid metric names are: {valid_metrics}")

    if fuzzy is True:
        if distance == "range":
            raise ValueError("The fuzzy option is not available for range distance.")

        # FuzzyEn: Remove the local baselines of vectors
        embedded -= np.mean(embedded, axis=1, keepdims=True)

        # TODO: it would be good to implement 'distrange' here to have fuzzy RangeEn

        dist_metric = sklearn.metrics.DistanceMetric.get_metric(distance)

        # Initialize count
        count = np.zeros(len(embedded))
        # Process in blocks
        block_size = 10
        for i in range(0, len(embedded), block_size):
            block = embedded[i : i + block_size]
            dist = dist_metric.pairwise(block, embedded)
            sim = member_func(func_name, dist, tolerance)
            count[i : i + block_size] = np.sum(sim, axis=1)

    elif distance == "range":
        # internal function for distrange
        def distrange(x, y):
            numerator = np.max(np.abs(x - y), axis=1) - np.min(np.abs(x - y), axis=1)
            denominator = np.max(np.abs(x - y), axis=1) + np.min(np.abs(x - y), axis=1)
            valid = np.where(denominator != 0)  # To prevent division by 0
            return np.divide(numerator[valid], denominator[valid])

        # Count for each row
        count = np.array([np.sum(distrange(embedded, embedded[i]) < tolerance) for i in range(len(embedded))])

    else:  # chebyshev and other sklearn methods
        # Perhaps scipy.spatial.KDTree would be faster? Especially since its query() method
        # has a `workers` argument to use multiple cores? Benchmark or opinion required!
        if kdtree is None:
            kdtree = sklearn.neighbors.KDTree(embedded, metric=distance)
        count = kdtree.query_radius(embedded, tolerance, count_only=True).astype(np.float64)
    return embedded, count, kdtree


# =============================================================================
# Membership Functions
# =============================================================================


def member_func(func_name, dist, tolerance):
    if func_name.lower() in membership_function:
        return membership_function[func_name.lower()](dist, tolerance)
    else:
        return (
            "Invalid function!\n"
            "Please choose one of the following:\n"
            "   exp    : exponential\n"
            "   gauss  : gaussian\n"
            "   cgauss : constgaussian\n"
            "   bell   : bell\n"
            "   z      : z\n"
            "   trapez   : trapezoidal\n"
            "   tri    : triangular\n"
            "   sig    : sigmoid\n"
        )


# see Azami et al. 2019
# https://doi.org/10.1109/access.2019.2930625
def exponential(dist, tolerance):
    # assert isinstance(tolerance,tuple), 'Tolerance must be a two-element tuple (threshold,power).'
    sim = np.exp(-(dist ** tolerance[1]) / tolerance[0])
    return sim


def gaussian(dist, tolerance):  # tolerance = sigma
    # assert np.size(tolerance)==1, 'Tolerance must be a scalar > 0.'
    sim = np.exp(-((dist**2) / (2 * (tolerance[0] ** 2))))
    return sim


def constgaussian(dist, tolerance):
    # assert np.size(tolerance)==1, 'Tolerance must be a scalar > 0.'
    sim = np.ones(np.shape(dist))
    sim[dist > tolerance[0]] = np.exp(-np.log(2) * ((dist[dist > tolerance[0]] - tolerance[0]) / tolerance[0]) ** 2)
    return sim


def bell(dist, tolerance):
    # assert isinstance(tolerance,tuple), 'Tolerance must be a two-element tuple (threshold,power).'
    sim = 1 / (1 + (abs(dist / tolerance[0]) ** (2 * tolerance[1])))
    return sim


def z(dist, tolerance):
    # assert np.size(tolerance)==1, 'Tolerance must be a scalar > 0.'
    sim = np.zeros(np.shape(dist))
    sim[dist <= 2 * tolerance[0]] = 2 * (((dist[dist <= 2 * tolerance[0]] - 2 * tolerance[0]) / tolerance[0]) ** 2)
    sim[dist <= 1.5 * tolerance[0]] = 1 - (
        2 * (((dist[dist <= 1.5 * tolerance[0]] - tolerance[0]) / tolerance[0]) ** 2)
    )
    sim[dist <= tolerance[0]] = 1
    return sim


def trapezoidal(dist, tolerance):
    # assert np.size(tolerance)==1, 'Tolerance must be a scalar > 0.'
    sim = np.zeros(np.shape(dist))
    sim[dist <= 2 * tolerance[0]] = -(dist[dist <= 2 * tolerance[0]] / tolerance[0]) + 2
    sim[dist <= tolerance[0]] = 1
    return sim


def triangular(dist, tolerance):
    # assert np.size(tolerance)==1, 'Tolerance must be a scalar > 0.'
    sim = 1 - (dist / tolerance[0])
    sim[dist > tolerance[0]] = 0
    return sim


def sigmoid(dist, tolerance):
    # see Zheng et al. 2018
    # https://doi.org/10.1016/j.measurement.2018.07.045
    # assert isinstance(tolerance,tuple), 'Tolerance must be a two-element tuple (a, threshold).'
    sim = 1 / (1 + np.exp(-tolerance[1](dist - tolerance[0])))
    return sim


membership_function = {
    "exp": exponential,
    "gauss": gaussian,
    "cgauss": constgaussian,
    "bell": bell,
    "z": z,
    "trapez": trapezoidal,
    "tri": triangular,
    "sig": sigmoid,
}
