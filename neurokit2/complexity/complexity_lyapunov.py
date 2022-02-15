# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise

from ..misc import NeuroKitWarning
from ..signal.signal_psd import signal_psd
from .complexity_embedding import complexity_embedding
from .optim_complexity_delay import complexity_delay


def complexity_lyapunov(
    signal,
    delay=1,
    dimension=2,
    method="rosenstein1993",
    len_trajectory=20,
    matrix_dim=4,
    min_neighbors="default",
    **kwargs,
):
    """(Largest) Lyapunov Exponent (LLE)

    Lyapunov exponents (LE) describe the rate of exponential separation (convergence or divergence)
    of nearby trajectories of a dynamical system. A system can have multiple LEs, equal to the
    number of the dimensionality of the phase space, and the largest LE value, `LLE` is often used to
    determine the overall predictability of the dynamical system.

    Different algorithms:

    - Rosenstein et al.'s (1993) algorithm was designed for calculating LLEs from small datasets.
      The time series is first reconstructed using a delay-embedding method, and the closest neighbour
      of each vector is computed using the euclidean distance. These two neighbouring points are then
      tracked along their distance trajectories for a number of data points. The slope of the line
      using a least-squares fit of the mean log trajectory of the distances gives the final LLE.

    - Eckmann et al. (1996) computes LEs by first reconstructing the time series using a
      delay-embedding method, and obtains the tangent that maps to the reconstructed dynamics using
      a least-squares fit, where the LEs are deduced from the tangent maps.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int, None
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common
        to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or
        to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
        If None for 'rosenstein1993', the delay is set to distance where the
        autocorrelation function drops below 1 - 1/e times its original value.
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns
        an array with two columns corresponding to the original signal and its delayed (by Tau) version.
        If method is 'eckmann1996', large values for dimension are recommended.
    method : str
        The method that defines the algorithm for computing LE. Can be one of 'rosenstein1993' or
        'eckmann1996'.
    len_trajectory : int
        The number of data points in which neighbouring trajectories are followed. Only relevant if
        method is 'rosenstein1993'.
    matrix_dim : int
        Correponds to the number of LEs to return for 'eckmann1996'.
    min_neighbors : int, str
        Minimum number of neighbors for 'eckmann1996'. If "default", min(2 * matrix_dim, matrix_dim + 4)
        is used.
    **kwargs : optional
        Other arguments to be passed to ``signal_psd()`` for calculating the minimum temporal
        separation of two neighbours.

    Returns
    --------
    lle : float
        An estimate of the largest Lyapunov exponent (LLE) if method is 'rosenstein1993', and
        an array of LEs if 'eckmann1996'.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute LLE.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=3, sampling_rate=100, frequency=[5, 8], noise=0.5)
    >>> lle, info = nk.complexity_lyapunov(signal, delay=1, dimension=2)
    >>> lle #doctest: +SKIP

    Reference
    ----------
    - Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993). A practical method
    for calculating largest Lyapunov exponents from small data sets.
    Physica D: Nonlinear Phenomena, 65(1-2), 117-134.

    - Eckmann, J. P., Kamphorst, S. O., Ruelle, D., & Ciliberto, S. (1986). Liapunov
    exponents from time series. Physical Review A, 34(6), 4971.
    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # If default tolerance
    tolerance = _complexity_lyapunov_tolerance(signal, **kwargs)  # rosenstein's method

    # Method
    method = method.lower()
    if method in ["rosenstein", "rosenstein1993"]:
        le, parameters = _complexity_lyapunov_rosenstein(
            signal, delay, dimension, tolerance, len_trajectory, **kwargs
        )
    elif method in ["eckmann", "eckmann1996"]:
        le, parameters = _complexity_lyapunov_eckmann(
            signal, delay, dimension, tolerance, matrix_dim, min_neighbors
        )

    # Store params
    info = {
        "Dimension": dimension,
        "Delay": delay,
        "Minimum Separation": tolerance,
        "Method": method,
    }
    info.update(parameters)

    return le, info


# =============================================================================
# Methods
# =============================================================================


def _complexity_lyapunov_rosenstein(
    signal, delay=1, dimension=2, tolerance=None, len_trajectory=20, **kwargs
):

    # If default tolerance (kwargs: tolerance="default")
    tolerance = _complexity_lyapunov_tolerance(signal, **kwargs)

    # Delay embedding
    if delay is None:
        delay = complexity_delay(signal, method="rosenstein1993", show=False)

    # Check that sufficient data points are available
    _complexity_lyapunov_checklength(
        len(signal), delay, dimension, tolerance, len_trajectory, method="rosenstein1993"
    )

    # Embed
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    m = len(embedded)

    # construct matrix with pairwise distances between vectors in orbit
    dists = sklearn.metrics.pairwise.euclidean_distances(embedded)
    for i in range(m):
        # Exclude indices within tolerance
        dists[i, max(0, i - tolerance) : i + tolerance + 1] = np.inf

    # Find indices of nearest neighbours
    ntraj = m - len_trajectory + 1
    min_dist_indices = np.argmin(dists[:ntraj, :ntraj], axis=1)  # exclude last few indices
    min_dist_indices = min_dist_indices.astype(int)

    # Follow trajectories of neighbour pairs for len_trajectory data points
    trajectories = np.zeros(len_trajectory)
    for k in range(len_trajectory):
        divergence = dists[(np.arange(ntraj) + k, min_dist_indices + k)]
        dist_nonzero = np.where(divergence != 0)[0]
        if len(dist_nonzero) == 0:
            trajectories[k] = -np.inf
        else:
            # Get average distances of neighbour pairs along the trajectory
            trajectories[k] = np.mean(np.log(divergence[dist_nonzero]))

    divergence_rate = trajectories[np.isfinite(trajectories)]

    # LLE obtained by least-squares fit to average line
    slope, _ = np.polyfit(np.arange(1, len(divergence_rate) + 1), divergence_rate, 1)

    parameters = {"Trajectory Length": len_trajectory}

    return slope, parameters


def _complexity_lyapunov_eckmann(
    signal, delay=1, dimension=2, tolerance=None, matrix_dim=4, min_neighbors="default"
):
    """TODO: check implementation

    From https://github.com/CSchoel/nolds
    """
    # Prepare parameters
    if min_neighbors == "default":
        min_neighbors = min(2 * matrix_dim, matrix_dim + 4)
    m = (dimension - 1) // (matrix_dim - 1)

    # Check that sufficient data points are available
    _complexity_lyapunov_checklength(
        len(signal),
        delay,
        dimension,
        tolerance,
        method="eckmann1996",
        matrix_dim=matrix_dim,
        min_neighbors=min_neighbors,
    )

    # Storing of LEs
    lexp = np.zeros(matrix_dim)
    lexp_counts = np.zeros(matrix_dim)
    old_Q = np.identity(matrix_dim)

    # Reconstruction using time-delay method
    embedded = complexity_embedding(signal[:-m], delay=delay, dimension=dimension)
    distances = sklearn.metrics.pairwise_distances(embedded, metric="chebyshev")

    for i in range(len(embedded)):
        # exclude difference of vector to itself and those too close in time
        distances[i, max(0, i - tolerance) : i + tolerance + 1] = np.inf

        # index of furthest nearest neighbour
        neighbour_furthest = np.argsort(distances[i])[min_neighbors - 1]

        # get neighbors within the radius
        r = distances[i][neighbour_furthest]
        neighbors = np.where(distances[i] <= r)[0]  # should have length = min_neighbours

        # Find matrix T_i (matrix_dim * matrix_dim) that sends points from neighbourhood of x(i) to x(i+1)
        vec_beta = signal[neighbors + matrix_dim * m] - signal[i + matrix_dim * m]
        matrix = np.array([signal[j : j + dimension : m] for j in neighbors])  # x(j)
        matrix -= signal[i : i + dimension : m]  # x(j) - x(i)

        # form matrix T_i
        t_i = np.zeros((matrix_dim, matrix_dim))
        t_i[:-1, 1:] = np.identity(matrix_dim - 1)
        t_i[-1] = np.linalg.lstsq(matrix, vec_beta, rcond=-1)[0]  # least squares solution

        # QR-decomposition of T * old_Q
        mat_Q, mat_R = np.linalg.qr(np.dot(t_i, old_Q))

        # force diagonal of R to be positive
        sign_diag = np.sign(np.diag(mat_R))
        sign_diag[np.where(sign_diag == 0)] = 1
        sign_diag = np.diag(sign_diag)
        mat_Q = np.dot(mat_Q, sign_diag)
        mat_R = np.dot(sign_diag, mat_R)

        old_Q = mat_Q
        # successively build sum for Lyapunov exponents
        diag_R = np.diag(mat_R)
        # filter zeros in mat_R (would lead to -infs)
        positive_elements = np.where(diag_R > 0)
        lexp_i = np.zeros(len(diag_R))
        lexp_i[positive_elements] = np.log(diag_R[positive_elements])
        lexp_i[np.where(diag_R == 0)] = np.inf

        lexp[positive_elements] += lexp_i[positive_elements]
        lexp_counts[positive_elements] += 1

    # normalize exponents over number of individual mat_Rs
    idx = np.where(lexp_counts > 0)
    lexp[idx] /= lexp_counts[idx]
    lexp[np.where(lexp_counts == 0)] = np.inf
    # normalize with respect to tau
    lexp /= delay
    # take m into account
    lexp /= m

    parameters = {"Minimum Neighbors": min_neighbors}

    return lexp, parameters


# =============================================================================
# Utilities
# =============================================================================


def _complexity_lyapunov_tolerance(signal, tolerance="default", **kwargs):
    """Minimum temporal separation (tolerance) between two neighbors.

    If 'default', finds a suitable value by calculating the mean period of the data,
    obtained by the reciprocal of the mean frequency of the power spectrum.

    https://github.com/CSchoel/nolds
    """
    if isinstance(tolerance, (int, float)):
        return tolerance

    psd = signal_psd(signal, sampling_rate=1000, method="fft", normalize=False, **kwargs)
    # actual sampling rate does not matter
    mean_freq = np.sum(psd["Power"] * psd["Frequency"]) / np.sum(psd["Power"])
    mean_period = 1 / mean_freq  # seconds per cycle
    tolerance = int(np.ceil(mean_period * 1000))

    return tolerance


def _complexity_lyapunov_checklength(
    n,
    delay=1,
    dimension=2,
    tolerance="default",
    len_trajectory=20,
    method="rosenstein1993",
    matrix_dim=4,
    min_neighbors="default",
):
    """Helper function that calculates the minimum number of data points required.
    """
    if method in ["rosenstein", "rosenstein1993"]:
        # minimum length required to find single orbit vector
        min_len = (dimension - 1) * delay + 1
        # we need len_trajectory orbit vectors to follow a complete trajectory
        min_len += len_trajectory - 1
        # we need tolerance * 2 + 1 orbit vectors to find neighbors for each
        min_len += tolerance * 2 + 1
        # Sanity check
        if n < min_len:
            warn(
                f"for dimension={dimension}, delay={delay}, tolerance={tolerance} and "
                + f"len_trajectory={len_trajectory}, you need at least {min_len} datapoints in your time series.",
                category=NeuroKitWarning,
            )

    elif method in ["eckmann", "eckmann1996"]:
        m = (dimension - 1) // (matrix_dim - 1)
        # minimum length required to find single orbit vector
        min_len = dimension
        # we need to follow each starting point of an orbit vector for m more steps
        min_len += m
        # we need tolerance * 2 + 1 orbit vectors to find neighbors for each
        min_len += tolerance * 2
        # we need at least min_nb neighbors for each orbit vector
        min_len += min_neighbors
        # Sanity check
        if n < min_len:
            warn(
                f"for dimension={dimension}, delay={delay}, tolerance={tolerance}, "
                + f"matrix_dim={matrix_dim} and min_neighbors={min_neighbors}, "
                + f"you need at least {min_len} datapoints in your time series.",
                category=NeuroKitWarning,
            )
