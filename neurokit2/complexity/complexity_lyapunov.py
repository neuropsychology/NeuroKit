# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise

from ..misc import NeuroKitWarning
from .complexity_embedding import complexity_embedding
from ..signal import signal_autocor


def _complexity_lyapunov_r(
    signal,
    delay=1,
    dimension=2,
    tau=1,
    min_neighbors=20,
    fit="RANSAC",
    debug_plot=False,
    debug_data=False,
    plot_file=None,
    fit_offset=0,
    **kwargs,
):
    """Lyapunov Exponents (LE) describe the rate of exponential separation (convergence or divergence)
    of nearby trajectories of a dynamical system. A system can have multiple LEs, equal to the number
    of the dimensionality of the phase space, and the largest LE value, `L1` is often used to determine
    the overall predictability of the dynamical system.

    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common
        to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or
        to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns
        an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    min_tsep : int | None
        Minimum temporal separation between two neighbors.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=3, sampling_rate=100, frequency=[5, 8], noise=0.5)
    >>> delay = 1; dimension = 2; complexity_embedding = nk.complexity_embedding; NeuroKitWarning=RuntimeWarning

    """
    
    # convert data to float to avoid overflow errors in rowwise_euclidean
    signal = np.asarray(signal, dtype=float)

    # Parameters
    n = len(signal)
    trajectory_len = 20

    # If default min_tsep (kwargs: min_tsep="default")
    min_tsep = _complexity_lyapunov_separation(signal, **kwargs)

    # Check that sufficient data points are available
    _complexity_lyapunov_checklength(n, delay, dimension, min_tsep, trajectory_len)

    # Delay embedding
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    m = len(embedded)

    # construct matrix with pairwise distances between vectors in orbit
    dists = sklearn.metrics.pairwise.euclidean_distances(embedded)

    # # we do not want to consider vectors as neighbor that are less than min_tsep
    # # time steps together => mask the distances min_tsep to the right and left of
    # # each index by setting them to infinity (will never be considered as nearest
    # # neighbors)
    # for i in range(m):
    #     dists[i, max(0, i - min_tsep) : i + min_tsep + 1] = np.inf
    # # check that we have enough data points to continue
    # ntraj = m - trajectory_len + 1
    # min_traj = min_tsep * 2 + 2  # in each row min_tsep + 1 disances are inf
    # if ntraj <= 0:
    #     msg = (
    #         "Not enough data points. Need {} additional data points to follow "
    #         + "a complete trajectory."
    #     )
    #     raise ValueError(msg.format(-ntraj + 1))
    # if ntraj < min_traj:
    #     # not enough data points => there are rows where all values are inf
    #     assert np.any(np.all(np.isinf(dists[:ntraj, :ntraj]), axis=1))
    #     msg = (
    #         "Not enough data points. At least {} trajectories are required "
    #         + "to find a valid neighbor for each orbit vector with min_tsep={} "
    #         + "but only {} could be created."
    #     )
    #     raise ValueError(msg.format(min_traj, min_tsep, ntraj))
    # assert np.all(np.any(np.isfinite(dists[:ntraj, :ntraj]), axis=1))
    # # find nearest neighbors (exclude last columns, because these vectors cannot
    # # be followed in time for trajectory_len steps)
    # nb_idx = np.argmin(dists[:ntraj, :ntraj], axis=1)

    # # build divergence trajectory by averaging distances along the trajectory
    # # over all neighbor pairs
    # div_traj = np.zeros(trajectory_len, dtype=float)
    # for k in range(trajectory_len):
    #     # calculate mean trajectory distance at step k
    #     indices = (np.arange(ntraj) + k, nb_idx + k)
    #     div_traj_k = dists[indices]
    #     # filter entries where distance is zero (would lead to -inf after log)
    #     nonzero = np.where(div_traj_k != 0)
    #     if len(nonzero[0]) == 0:
    #         # if all entries where zero, we have to use -inf
    #         div_traj[k] = -np.inf
    #     else:
    #         div_traj[k] = np.mean(np.log(div_traj_k[nonzero]))
    # # filter -inf entries from mean trajectory
    # ks = np.arange(trajectory_len)
    # finite = np.where(np.isfinite(div_traj))
    # ks = ks[finite]
    # div_traj = div_traj[finite]
    # if len(ks) < 1:
    #     # if all points or all but one point in the trajectory is -inf, we cannot
    #     # fit a line through the remaining points => return -inf as exponent
    #     poly = [-np.inf, 0]
    # else:
    #     # normal line fitting
    #     poly = poly_fit(ks[fit_offset:], div_traj[fit_offset:], 1, fit=fit)
    # if debug_plot:
    #     plot_reg(ks[fit_offset:], div_traj[fit_offset:], poly, "k", "log(d(k))", fname=plot_file)
    # le = poly[0] / tau
    # if debug_data:
    #     return (le, (ks, div_traj, poly))
    # else:
    #     return le


# =============================================================================
# Utilities
# =============================================================================
def _complexity_lyapunov_delay(signal):
    """Compute optimal lag as the point where the autocorrelation function drops
    to (1 − 1 / e) of its initial value, according to Rosenstein et al. (1993).
    
    Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993).
    A practical method for calculating largest Lyapunov exponents from small data sets.
    Physica D: Nonlinear Phenomena, 65(1-2), 117-134.
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
            f"Signal has a mean frequency too low for min_tsep={min_tsep}, setting min_tsep={int(max_tsep_factor * n)}",
            category=NeuroKitWarning,
        )
        min_tsep = int(max_tsep_factor * n)
    return min_tsep


def _complexity_lyapunov_checklength(
    n, delay=1, dimension=2, min_tsep="default", trajectory_len=20
):
    """
    Helper function that calculates the minimum number of data points required.

    trajectory_len is the time (in number of data points) to follow the distance trajectories
    between two neighboring points
    """
    # minimum length required to find single orbit vector
    min_len = (dimension - 1) * delay + 1
    # we need trajectory_len orbit vectors to follow a complete trajectory
    min_len += trajectory_len - 1
    # we need min_tsep * 2 + 1 orbit vectors to find neighbors for each
    min_len += min_tsep * 2 + 1

    # Sanity check
    if n < min_len:
        warn(
            f"for dimension={dimension}, delay={delay}, min_tsep={min_tsep} and trajectory_len={trajectory_len}, you need at least {min_len} datapoints in your time series",
            category=NeuroKitWarning,
        )
