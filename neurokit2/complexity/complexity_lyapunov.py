# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise

from ..misc import NeuroKitWarning
from .complexity_embedding import complexity_embedding
from ..signal import signal_autocor


def complexity_lyapunov(
    signal,
    delay=1,
    dimension=2,
    len_trajectory=20,
    min_tsep=None,
    **kwargs,
):
    """Lyapunov Exponents (LE) describe the rate of exponential separation (convergence or divergence)
    of nearby trajectories of a dynamical system. A system can have multiple LEs, equal to the number
    of the dimensionality of the phase space, and the largest LE value, `L1` is often used to determine
    the overall predictability of the dynamical system.

    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int, None
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common
        to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or
        to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
        If None, the delay is set to distance where the autocorrelation function drops below 1 - 1/e times
        its original value.
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns
        an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    len_trajectory : int
        The number of data points in which neighbouring trajectories are followed.
    min_tsep : int, bool
        Minimum temporal separation between two neighbors. If None (default), `min_tsep` is set to the mean
        period of the signal obtained by computing the mean frequency using the fast fourier transform.
    **kwargs : optional
        Other arguments.

    Returns
    --------
    l1 : float
        An estimate of the largest Lyapunov exponent (LLE).
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute LLE.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=3, sampling_rate=100, frequency=[5, 8], noise=0.5)
    >>> l1, _ = nk.complexity_lyapunov(signal, delay=1, dimension=2)
    >>> l1 #doctest: +SKIP

    Reference
    ----------
    - Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993). A practical method
    for calculating largest Lyapunov exponents from small data sets.
    Physica D: Nonlinear Phenomena, 65(1-2), 117-134.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )
    n = len(signal)

    # If default min_tsep (kwargs: min_tsep="default")
    min_tsep = _complexity_lyapunov_separation(signal, **kwargs)

    # Check that sufficient data points are available
    _complexity_lyapunov_checklength(n, delay, dimension, min_tsep, len_trajectory)

    # Delay embedding
    if delay is None:
        delay = _complexity_lyapunov_delay(signal)
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    m = len(embedded)
    info = {"Dimension": dimension, "Delay": delay, "Minimum Separation": min_tsep}

    # construct matrix with pairwise distances between vectors in orbit
    dists = sklearn.metrics.pairwise.euclidean_distances(embedded)

    min_dist = np.zeros(m)
    min_dist_indices = np.zeros(m)
    for i in range(m):
        # Exclude indices within min_tsep
        dists[i, max(0, i - min_tsep) : i + min_tsep + 1] = np.inf

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

    return slope, info

# =============================================================================
# Utilities
# =============================================================================
def _complexity_lyapunov_delay(signal):
    """Compute optimal lag as the point where the autocorrelation function drops
    to (1 − 1 / e) of its initial value, according to Rosenstein et al. (1993).
    
    """
    # not sure if this is better to be in `optim_complexity_delay` or if this is specific
    # only for lyapunov
    threshold = 1 - 1 / np.e
    delay = np.where(signal_autocor(signal, method='fft')[0] < threshold)[0][0]

    return delay


def _complexity_lyapunov_separation(signal, min_tsep="default"):
    """Minimum temporal separation between two neighbors.

    If 'default', finds a suitable value by calculating the mean period of the data.

    https://github.com/CSchoel/nolds
    """
    if isinstance(min_tsep, (int, float)):
        return min_tsep

    n = len(signal)
    max_tsep_factor = 0.25

    # min_tsep need the fft
    f = np.fft.rfft(signal, n * 2 - 1)
    # calculate min_tsep as mean period (= 1 / mean frequency)
    mf = np.fft.rfftfreq(n * 2 - 1) * np.abs(f)
    mf = np.mean(mf[1:]) / np.sum(np.abs(f[1:]))
    min_tsep = int(np.ceil(1.0 / mf))
    if min_tsep > max_tsep_factor * n:
        warn(
            f"Signal has a mean frequency too low for min_tsep={min_tsep}, " + 
            f"setting min_tsep={int(max_tsep_factor * n)}",
            category=NeuroKitWarning,
        )
        min_tsep = int(max_tsep_factor * n)
    return min_tsep


def _complexity_lyapunov_checklength(
    n, delay=1, dimension=2, min_tsep="default", len_trajectory=20
):
    """
    Helper function that calculates the minimum number of data points required.
    """
    # minimum length required to find single orbit vector
    min_len = (dimension - 1) * delay + 1
    # we need len_trajectory orbit vectors to follow a complete trajectory
    min_len += len_trajectory - 1
    # we need min_tsep * 2 + 1 orbit vectors to find neighbors for each
    min_len += min_tsep * 2 + 1

    # Sanity check
    if n < min_len:
        warn(
            f"for dimension={dimension}, delay={delay}, min_tsep={min_tsep} and " +
            f"len_trajectory={len_trajectory}, you need at least {min_len} datapoints in your time series",
            category=NeuroKitWarning,
        )
