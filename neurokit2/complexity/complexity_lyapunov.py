from collections import deque
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial

from ..misc import NeuroKitWarning
from .utils_complexity_embedding import complexity_embedding


def complexity_lyapunov(
    signal,
    delay=1,
    dimension=2,
    method="makowski",
    separation="auto",
    sampling_rate=None,
    show=False,
    **kwargs,
):
    """**(Largest) Lyapunov Exponent (LLE)**

    Lyapunov exponents (LE) describe the rate of exponential separation (convergence or divergence)
    of nearby trajectories of a dynamical system. It is a measure of sensitive dependence on
    initial conditions, i.e. how quickly two nearby states diverge. A system can have multiple LEs,
    equal to the number of the dimensionality of the phase space, and the largest LE value, "LLE" is
    often used to determine the overall predictability of the dynamical system. A positive LLE is
    an indicator of chaos (although the estimates are also positive for noise), while it is 0 for a
    periodic signal.

    Different algorithms exist to estimate these indices:

    * **Rosenstein et al.'s (1993)** algorithm was designed for calculating LLEs from small
      datasets. The time series is first reconstructed using a delay-embedding method, and the
      closest neighbour :math:`X_{\\hat{j}}` of each vector :math:`X_j` is found using the
      euclidean distance, among the vectors that are more than ``separation`` samples away in time
      (by default, the mean period of the signal). These two neighbouring points are then tracked
      along their trajectories for ``len_trajectory`` samples, and the average logarithm of their
      distances :math:`d_j(i) = \\| X_{j + i} - X_{\\hat{j} + i} \\|` gives the divergence curve
      (the pairs at a distance of 0 are left out):

      .. math::

        y(i) = \\langle \\ln d_j(i) \\rangle_j

      The LLE is the slope of the least-squares line fitted to this curve, from :math:`i = 0` to
      ``len_trajectory - 1``. The fit should only cover the initial linear part of the curve,
      before the distances saturate at the size of the attractor (e.g., within about 5 to 10
      iterations for chaotic maps), so ``len_trajectory`` should be adjusted accordingly (see
      ``show=True``): with the default of 20 samples, the LLE of the logistic map (ln(2) = 0.69)
      is estimated at 0.35, and at 0.69 with ``len_trajectory=8``. The linear part grows with the
      length of the signal and with its sampling rate, and for flows (e.g., the Lorenz system),
      the curve starts with a transient steeper than the LLE (about 2.5 times the LLE over the
      first 20 samples, at 100 samples per time unit), so that no fixed ``len_trajectory`` suits
      all signals. The ``"makowski"`` method chooses the fit range automatically.
    * **Makowski** (default) is a modification of Rosenstein's algorithm, in which the part of the
      divergence curve to which the line is fitted is chosen automatically. The nearest neighbours
      are found as in Rosenstein's method, and at each step, all the pairs that can still be
      followed are used. They are followed until the curve levels off, i.e., until it reaches the
      level of unrelated vectors (the mean log distance between random pairs of vectors more than
      ``separation`` samples apart) minus 0.25, or until it rises by less than 0.1 over
      ``separation`` steps (e.g., for quasi-periodic signals, whose neighbours stay closer than
      unrelated vectors), or for at most ``max_length`` steps. The local slope of the curve is then
      computed at each step (the slope of the least-squares line over ``separation / 4`` steps
      on each side, which smooths out its oscillations within each period), from :math:`i = 1`
      (the distance at :math:`i = 0` is a minimum, biased low by the choice of the nearest
      neighbour) up to the first step within 0.5 of the level at which the curve levels off. The
      LLE is the slope of the least-squares line over the first run of steps in which the local
      slope is positive and varies by less than a factor of 1.5 that is at least 80% as long as
      the longest such run (the curve bends gradually towards its plateau, so that the earliest
      of the long runs is the least affected by the saturation). This leaves out the initial
      transient of flows and the saturation, so that the LLE does not depend on the length of the
      signal or on its sampling rate (e.g., 0.98 to 1.00 times the LLE of the logistic, tent and
      Henon maps, and 0.94 and 0.87 times that of the Lorenz and Rossler systems, against 0.43 to
      0.81, 2.5 and 1.7 with Rosenstein's default of 20 samples). If there is no such run (with a
      warning), the LLE is the slope of the curve from :math:`i = 1` if the curve levels off more
      than 0.5 below the level of unrelated vectors (the trajectories do not diverge, e.g., for
      periodic signals, with or without noise), and NaN otherwise (the neighbours become unrelated
      within a few steps, e.g., for white noise).
    * **Eckmann et al. (1986)** estimates the spectrum of LEs from the tangent maps of the
      reconstructed dynamics. The neighbours of each vector :math:`X_i` of the delay-embedding
      (of ``dimension`` :math:`d_E`) are the vectors within the (Chebyshev) distance of its
      ``min_neighbors``-th nearest neighbour, among those that are more than ``separation`` samples
      away in time. The local linear map :math:`T_i`, of dimension ``matrix_dim`` :math:`d_M`,
      that sends the neighbourhood of :math:`X_i` to that of its future, is then fitted by least
      squares on the vectors :math:`(x_j, x_{j + m\\tau}, ..., x_{j + (d_M - 1)m\\tau})`, where
      :math:`\\tau` is the ``delay`` (1 in the original method) and :math:`m = (d_E - 1) / (d_M - 1)`
      (so that :math:`d_E - 1` must be a multiple of :math:`d_M - 1`):

      .. math::

        T_i (X_j - X_i) \\approx X_{j + m\\tau} - X_{i + m\\tau}

      The tangent maps are multiplied along the trajectory using QR decompositions,
      :math:`T_i Q_{i - m\\tau} = Q_i R_i`, and the LEs are the average logarithms of the diagonal
      elements of the :math:`R_i`, divided by the time step :math:`m\\tau`:

      .. math::

        \\lambda_k = \\frac{1}{m\\tau} \\langle \\ln (R_i)_{kk} \\rangle_i

    .. note::

      Observational noise makes the divergence curve jump in the first step and flattens it
      between this noise floor and its plateau, so that the LLE of noisy signals is
      underestimated (e.g., by about 25% for chaotic maps and the Lorenz system with noise of 5% of
      their standard deviation, with ``"makowski"``), as with any method fitting the linear part of
      the curve. Smooth noise (e.g., an autoregressive process) can also give a positive LLE: a
      positive LLE is evidence of chaos only if it is larger than that of surrogate data (e.g.,
      with the same power spectrum).

      With Eckmann's method, if ``matrix_dim`` is larger than the dimension of the attractor, the
      spectrum also contains spurious exponents, which can be larger than the true LLE (e.g.,
      multiples of the LLE, such as 1.39 and 0.69 for the logistic map with ``dimension=2``;
      Sauer et al., 1998). For noise-free signals that repeat themselves exactly (e.g., a sine
      whose period is a whole number of samples), the neighbours are copies of each other, at
      distances of the order of the rounding errors, so that the estimates are meaningless. Overall,
      we would like to improve this function to implement newer and faster methods (e.g.,
      Balcerzak, 2018, 2020), etc. If you're interested in helping out with this, please get in
      touch!

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter. If method
        is ``"eckmann1986"``, it is the dimension :math:`d_E` in which the neighbours are found, and
        larger values are recommended.
    method : str
        The method that defines the algorithm for computing LE. Can be one of ``"makowski"``
        (default), ``"rosenstein1993"``, or ``"eckmann1986"``.
    separation : int, str
        Minimum temporal separation (in samples) between two neighbours (also known as the Theiler
        window): only the vectors that are more than ``separation`` samples apart can be
        neighbours. If ``"auto"`` (default), it is the mean period of the signal (rounded up), i.e.,
        the reciprocal of the mean frequency of its power spectrum, as in Rosenstein et al. (1993).
    sampling_rate : int, float
        The sampling rate of the signal (in Hz, i.e., samples/second). If given, the exponents are
        expressed per second instead of per sample.
    show : bool
        If ``True``, plots the divergence curve and the fitted line (with the fit range and the
        level at which the curve levels off for ``"makowski"``; not available for
        ``"eckmann1986"``).
    len_trajectory : int
        Applies when method is ``"rosenstein1993"``. The number of data points in which
        neighboring trajectories are followed (i.e., the number of points of the divergence curve to
        which the line is fitted). Defaults to 20.
    max_length : int, str
        Applies when method is ``"makowski"``. The maximum number of steps over which neighboring
        trajectories are followed (they are followed until the divergence curve levels off, which
        is usually earlier). If ``"auto"`` (default), 20 times the ``separation``. It is capped at
        half the number of embedded vectors.
    matrix_dim : int, str
        Applies when method is ``"eckmann1986"``. The dimension :math:`d_M` of the tangent maps,
        which corresponds to the number of LEs to return. ``dimension - 1`` must be a multiple of
        ``matrix_dim - 1``. If ``"default"``, ``min(dimension, 4)`` is used.
    min_neighbors : int, str
        Applies when method is ``"eckmann1986"``. Minimum number of neighbors. If ``"default"``,
        ``min(2 * matrix_dim, matrix_dim + 4)`` is used.
    **kwargs : optional
        The method-specific arguments listed above (``len_trajectory``, ``max_length``,
        ``matrix_dim`` and ``min_neighbors``), which are ignored by the other methods.

    Returns
    --------
    lle : float
        An estimate of the largest Lyapunov exponent (LLE) if method is ``"rosenstein1993"`` or
        ``"makowski"``, and an array of LEs if ``"eckmann1986"``, per sample (in natural logarithm
        units), or per second if ``sampling_rate`` is given. It is NaN if the signal is constant or
        has missing values, and with ``"makowski"``, if the divergence curve has no linear part
        (e.g., for white noise).
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute LLE, such as the divergence curve (``"Divergence_Rate"``, against the number of
        steps, i.e., samples, from 0) and the intercept of the fitted line (``"Intercept"``) for
        ``"rosenstein1993"`` and ``"makowski"``. For ``"makowski"``, it also contains the first and
        last steps to which the line is fitted (``"Fit_Range"``), the :math:`R^2` of the fit (``"R2"``),
        the level at which the curve levels off (``"Plateau"``), and the number of pairs of
        neighbours followed at each step (``"Pairs"``).

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Lorenz system, 20 time units per second (its LLE is 0.906 per time unit, i.e., 18.1 per
      # second here)
      signal = nk.complexity_simulate(duration=5, sampling_rate=1000, method="lorenz")

      # Makowski's method (default), per sample
      @savefig p_complexity_lyapunov1.png scale=100%
      lle, info = nk.complexity_lyapunov(signal, delay=5, dimension=3, show=True)
      @suppress
      plt.close()

      lle
      info["Fit_Range"]

      # Per second rather than per sample
      lle, info = nk.complexity_lyapunov(signal, delay=5, dimension=3, sampling_rate=1000)
      lle

      # Rosenstein's method (fitted over the first 20 samples)
      @savefig p_complexity_lyapunov2.png scale=100%
      lle, info = nk.complexity_lyapunov(
          signal, delay=5, dimension=3, method="rosenstein1993", sampling_rate=1000, show=True
      )
      @suppress
      plt.close()

      lle

      # Eckmann's method (spectrum of LEs)
      les, info = nk.complexity_lyapunov(signal, delay=5, dimension=3, method="eckmann1986", sampling_rate=1000)
      les

    References
    ----------
    * Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993). A practical method
      for calculating largest Lyapunov exponents from small data sets.
      Physica D: Nonlinear Phenomena, 65(1-2), 117-134.
    * Eckmann, J. P., Kamphorst, S. O., Ruelle, D., & Ciliberto, S. (1986). Liapunov
      exponents from time series. Physical Review A, 34(6), 4971.
    * Sauer, T. D., Tempkin, J. A., & Yorke, J. A. (1998). Spurious Lyapunov exponents in
      attractor reconstruction. Physical Review Letters, 81(20), 4341.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    signal = np.asarray(signal, dtype=float)

    # Method, and its specific arguments (the arguments of the other methods are ignored)
    method = method.lower()
    if method in ["rosenstein", "rosenstein1993"]:
        method_function, arguments = _complexity_lyapunov_rosenstein, ["len_trajectory"]
    elif method in ["makowski"]:
        method_function, arguments = _complexity_lyapunov_makowski, ["max_length"]
    elif method in ["eckmann", "eckmann1986"]:
        method_function, arguments = _complexity_lyapunov_eckmann, ["matrix_dim", "min_neighbors"]
    else:
        raise ValueError(
            "NeuroKit error: complexity_lyapunov(): 'method' should be one of 'makowski', 'rosenstein1993', 'eckmann1986'."
        )
    unknown = [k for k in kwargs if k not in ["len_trajectory", "max_length", "matrix_dim", "min_neighbors"]]
    if len(unknown) > 0:
        raise TypeError(f"NeuroKit error: complexity_lyapunov(): unexpected argument(s) {unknown}.")
    kwargs = {k: v for k, v in kwargs.items() if k in arguments}

    # Store params
    info = {"Dimension": dimension, "Delay": delay, "Separation": separation, "Method": method}

    # The LE is undefined for constant signals or signals with missing values
    if signal.size == 0 or np.any(~np.isfinite(signal)) or np.ptp(signal) == 0:
        warn(
            "The signal is constant or contains missing (or infinite) values. The Lyapunov exponent "
            + "cannot be computed and is returned as NaN.",
            category=NeuroKitWarning,
        )
        if method_function is _complexity_lyapunov_eckmann:
            return np.full(_complexity_lyapunov_matrixdim(dimension, kwargs.get("matrix_dim", "default")), np.nan), info
        return np.nan, info

    # Compute Minimum temporal separation between two neighbors
    # -----------------------------------------------------------
    # Rosenstein (1993) finds a suitable value by calculating the mean period of the data,
    # obtained by the reciprocal of the mean frequency of the power spectrum.

    # "We impose the additional constraint that nearest neighbors have a temporal separation
    # greater than the mean period of the time series: This allows us to consider each pair of
    # neighbors as nearby initial conditions for different trajectories."

    # "We estimated the mean period as the reciprocal of the mean frequency of the power spectrum,
    # although we expect any comparable estimate, e.g., using the median frequency of the magnitude
    # spectrum, to yield equivalent results."
    if isinstance(separation, str) and separation == "auto":
        # Power spectrum of the signal (without its mean), with the frequencies in cycles per
        # sample, so that the mean period is in samples
        power = np.abs(np.fft.rfft(signal - np.mean(signal))) ** 2
        frequency = np.fft.rfftfreq(len(signal))
        mean_frequency = np.sum(power[1:] * frequency[1:]) / np.sum(power[1:])
        separation = int(np.ceil(1 / mean_frequency))
    elif isinstance(separation, (int, np.integer)) and not isinstance(separation, bool) and separation >= 0:
        separation = int(separation)
    else:
        raise ValueError("NeuroKit error: complexity_lyapunov(): 'separation' should be 'auto' or a non-negative integer.")
    info["Separation"] = separation

    # Run algorithm
    # ----------------
    le, parameters = method_function(signal, delay=delay, dimension=dimension, separation=separation, **kwargs)
    info.update(parameters)

    # Exponents per second rather than per sample
    if sampling_rate is not None:
        le = le * sampling_rate

    if show is True and method_function is not _complexity_lyapunov_eckmann:
        _complexity_lyapunov_plot(le, info, sampling_rate)

    return le, info


# =============================================================================
# Methods
# =============================================================================
def _complexity_lyapunov_makowski(
    signal,
    delay=1,
    dimension=2,
    separation=1,
    max_length="auto",
):
    # Embedding
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    n = len(embedded)
    _complexity_lyapunov_checklength(n, separation)

    # Follow the trajectories for at most 20 times the separation (the mean period by default), and
    # half the number of vectors
    if isinstance(max_length, str) and max_length == "auto":
        max_length = max(20 * separation, 20)
    if not isinstance(max_length, (int, np.integer)) or isinstance(max_length, bool) or max_length < 2:
        raise ValueError("NeuroKit error: complexity_lyapunov(): 'max_length' should be 'auto' or an integer of at least 2.")
    max_length = int(min(max_length, n // 2))

    # Nearest neighbour of each vector, outside of the `separation` region
    idx = _complexity_lyapunov_neighbours(embedded, separation)

    # Level of the distances between unrelated vectors: mean log distance between random pairs of
    # vectors more than `separation` samples apart (with a fixed seed, so that the result is
    # deterministic)
    pairs = np.random.default_rng(0).integers(0, n, size=(2, 10000))
    pairs = pairs[:, np.abs(pairs[0] - pairs[1]) > separation]
    distances = np.linalg.norm(embedded[pairs[0]] - embedded[pairs[1]], axis=1)
    distances = distances[distances > 0]
    level = np.mean(np.log(distances)) if len(distances) > 0 else np.inf

    # Average divergence at each step, with all the pairs that can still be followed, until the
    # curve levels off: when it reaches the level of unrelated vectors (chaotic signals), or when
    # it rises by less than 0.1 over `separation` steps (e.g., quasi-periodic signals, whose
    # neighbours stay closer than unrelated vectors)
    block = max(separation, 5)
    plateau = level
    trajectories, n_pairs = [], []
    for k in range(max_length + 1):
        valid = np.where((np.arange(n) + k < n) & (idx + k < n))[0]
        divergences = np.linalg.norm(embedded[valid + k] - embedded[idx[valid] + k], axis=1)
        divergences = divergences[divergences > 0]
        trajectories.append(np.mean(np.log(divergences)) if len(divergences) > 0 else np.nan)
        n_pairs.append(len(divergences))
        if trajectories[-1] >= level - 0.25:
            break
        if (k + 1) % block == 0 and k + 1 >= 2 * block:
            last, previous = _mean_finite(trajectories[-block:]), _mean_finite(trajectories[-2 * block : -block])
            if last - previous <= 0.1:
                # Its plateau is then its highest running mean
                running = np.convolve(trajectories, np.ones(block) / block, mode="valid")
                running = running[np.isfinite(running)]
                if len(running) > 0:
                    plateau = min(level, np.max(running))
                break
    trajectories = np.array(trajectories)
    info = {"Divergence_Rate": trajectories, "Pairs": np.array(n_pairs)}

    if np.sum(np.isfinite(trajectories)) < 2:
        warn(
            "The distances between neighbours are 0 along the trajectories. The Lyapunov exponent "
            + "cannot be estimated and is returned as NaN.",
            category=NeuroKitWarning,
        )
        info.update({"Plateau": np.nan, "Fit_Range": (0, len(trajectories) - 1), "R2": np.nan, "Intercept": np.nan})
        return np.nan, info
    if not np.isfinite(plateau):
        plateau = np.nanmax(trajectories)
    info["Plateau"] = plateau

    # Rise of the curve: from the first step (the distance at step 0 is a minimum, biased low by
    # the choice of the nearest neighbour) to the first step within 0.5 of its plateau
    above = np.where(~(trajectories[1:] < plateau - 0.5))[0]
    stop = 1 + above[0] if len(above) > 0 else len(trajectories) - 1

    # Linear part: the first of the longest runs of steps in which the local slope (over
    # separation / 4 steps on each side, which smooths out its oscillations within each period)
    # varies by less than a factor 1.5
    slopes = _complexity_lyapunov_localslopes(trajectories[1 : stop + 1], max(1, int(round(separation / 4))))
    start, end = _complexity_lyapunov_linearrun(slopes, ratio=1.5, share=0.8)
    start, end = start + 1, end + 1

    estimable = True
    if end - start < 2:
        if plateau < level - 0.5:
            # The trajectories stay closer than unrelated vectors: they do not diverge (apart from
            # the noise, in the first step), e.g., periodic signals
            warn(
                "The divergence curve has no linear part: the distances between neighbours do not grow "
                + "(apart from the first step), as for periodic signals. The Lyapunov exponent is the slope "
                + "of the curve from the first step (close to 0).",
                category=NeuroKitWarning,
            )
            start, end = min(1, len(trajectories) - 2), len(trajectories) - 1
        else:
            warn(
                "The divergence curve has no linear part: the distances between neighbours reach their "
                + f"plateau within {stop} steps, as for noise (or signals dominated by noise). The Lyapunov "
                + "exponent cannot be estimated and is returned as NaN.",
                category=NeuroKitWarning,
            )
            start, end, estimable = 0, stop, False
    info["Fit_Range"] = (int(start), int(end))

    # Linear fit
    steps = np.arange(start, end + 1)
    curve = trajectories[start : end + 1]
    finite = np.isfinite(curve)
    if not estimable or np.sum(finite) < 2:
        slope, intercept, r2 = np.nan, np.nan, np.nan
    else:
        slope, intercept = np.polyfit(steps[finite], curve[finite], 1)
        residuals = curve[finite] - (intercept + slope * steps[finite])
        total = np.sum((curve[finite] - np.mean(curve[finite])) ** 2)
        r2 = 1 - np.sum(residuals**2) / total if total > 0 else np.nan
    info.update({"R2": r2, "Intercept": intercept})
    return slope, info


def _complexity_lyapunov_rosenstein(signal, delay=1, dimension=2, separation=1, len_trajectory=20):
    if not isinstance(len_trajectory, (int, np.integer)) or len_trajectory < 2:
        raise ValueError("NeuroKit error: complexity_lyapunov(): 'len_trajectory' should be an integer of at least 2.")

    # Embedding
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)

    # Only the vectors that can be followed for len_trajectory data points are used, both as
    # reference points and as neighbours
    ntraj = len(embedded) - len_trajectory + 1
    _complexity_lyapunov_checklength(
        ntraj,
        separation,
        details=f"for dimension={dimension}, delay={delay}, separation={separation} and len_trajectory={len_trajectory}, "
        + f"at least {(dimension - 1) * delay + len_trajectory + 2 * separation + 1} data points are needed "
        + f"({len(signal)} given).",
    )

    # Find indices of nearest neighbours, more than `separation` samples apart
    min_dist_indices = _complexity_lyapunov_neighbours(embedded[:ntraj], separation)

    # Follow trajectories of neighbour pairs for len_trajectory data points
    trajectories = np.zeros(len_trajectory)
    for k in range(len_trajectory):
        divergence = np.linalg.norm(embedded[np.arange(ntraj) + k] - embedded[min_dist_indices + k], axis=1)
        dist_nonzero = divergence[divergence != 0]
        if len(dist_nonzero) == 0:
            trajectories[k] = -np.inf
        else:
            # Get average distances of neighbour pairs along the trajectory
            trajectories[k] = np.mean(np.log(dist_nonzero))

    # LLE obtained by least-squares fit to average line (at the steps where it is defined)
    steps = np.arange(len_trajectory)
    finite = np.isfinite(trajectories)
    if np.sum(finite) < 2:
        warn(
            "The distances between neighbours are 0 along the trajectories. The Lyapunov exponent "
            + "cannot be computed and is returned as NaN.",
            category=NeuroKitWarning,
        )
        slope, intercept = np.nan, np.nan
    else:
        slope, intercept = np.polyfit(steps[finite], trajectories[finite], 1)

    # Store info
    parameters = {
        "Trajectory_Length": len_trajectory,
        "Divergence_Rate": trajectories,
        "Intercept": intercept,
    }

    return slope, parameters


def _complexity_lyapunov_eckmann(
    signal,
    delay=1,
    dimension=2,
    separation=0,
    matrix_dim="default",
    min_neighbors="default",
):
    """Adapted from https://github.com/CSchoel/nolds (lyap_e)."""
    # Prepare parameters
    matrix_dim = _complexity_lyapunov_matrixdim(dimension, matrix_dim)
    if isinstance(min_neighbors, str) and min_neighbors == "default":
        min_neighbors = min(2 * matrix_dim, matrix_dim + 4)
    if min_neighbors < 1:
        raise ValueError("NeuroKit error: complexity_lyapunov(): 'min_neighbors' should be at least 1.")
    m = (dimension - 1) // (matrix_dim - 1)
    # The tangent maps send the vectors to their future, `step` samples ahead
    step = m * delay

    # Reconstruction using time-delay method. We need to be able to step `step` points further
    # for the beta vector, i.e., matrix_dim * step points after the start of each vector
    ntraj = len(signal) - matrix_dim * step
    _complexity_lyapunov_checklength(
        ntraj,
        separation,
        n_neighbours=min_neighbors,
        details=f"for dimension={dimension}, delay={delay}, separation={separation}, matrix_dim={matrix_dim} "
        + f"and min_neighbors={min_neighbors}, at least {matrix_dim * step + 2 * separation + min_neighbors + 1} "
        + f"data points are needed ({len(signal)} given).",
    )
    embedded = complexity_embedding(signal[: ntraj + (dimension - 1) * delay], delay=delay, dimension=dimension)

    # Neighbours within the radius of the min_neighbors-th nearest neighbour (Chebyshev distance)
    neighbours = _complexity_lyapunov_neighbours(embedded, separation, n_neighbours=min_neighbors, p=np.inf)

    # Storing of LEs
    lexp = np.zeros(matrix_dim)
    lexp_counts = np.zeros(matrix_dim)
    # The tangent maps are multiplied along the trajectory, every `step` samples (T_1, T_1+m,
    # T_1+2m, ... in Eckmann et al., 1986), so that all the vectors are used in `step` interleaved
    # products (each starting from the identity)
    old_Q = [np.identity(matrix_dim) for _ in range(step)]
    # Matrix T_i (matrix_dim * matrix_dim) that sends points from neighbourhood of x(i) to x(i+step)
    t_i = np.zeros((matrix_dim, matrix_dim))
    t_i[:-1, 1:] = np.identity(matrix_dim - 1)
    lags = np.arange(matrix_dim) * step

    for i in range(ntraj):
        j = neighbours[i]

        # Least squares solution for the last row of T_i
        vec_beta = signal[j + matrix_dim * step] - signal[i + matrix_dim * step]
        matrix = signal[j[:, np.newaxis] + lags] - signal[i + lags]  # x(j) - x(i)
        t_i[-1] = np.linalg.lstsq(matrix, vec_beta, rcond=-1)[0]

        # QR-decomposition of T * old_Q
        mat_Q, mat_R = np.linalg.qr(np.dot(t_i, old_Q[i % step]))

        # force diagonal of R to be positive
        sign_diag = np.sign(np.diag(mat_R))
        sign_diag[sign_diag == 0] = 1
        old_Q[i % step] = mat_Q * sign_diag

        # successively build sum for Lyapunov exponents
        diag_R = sign_diag * np.diag(mat_R)
        # filter zeros in mat_R (would lead to -infs)
        positive_elements = diag_R > 0
        lexp[positive_elements] += np.log(diag_R[positive_elements])
        lexp_counts[positive_elements] += 1

    # normalize exponents over number of individual mat_Rs (the exponent is -inf if all the
    # diagonal elements were 0)
    idx = lexp_counts > 0
    lexp[idx] /= lexp_counts[idx]
    lexp[~idx] = -np.inf
    # normalize with respect to the time step (in samples)
    lexp /= step

    parameters = {"Matrix_Dimension": matrix_dim, "Minimum_Neighbors": min_neighbors}

    return lexp, parameters


# =============================================================================
# Utilities
# =============================================================================
def _complexity_lyapunov_matrixdim(dimension, matrix_dim="default"):
    """Check the dimension of the tangent maps of Eckmann's method."""
    if isinstance(matrix_dim, str) and matrix_dim == "default":
        matrix_dim = min(dimension, 4)
    valid = [d for d in range(2, int(dimension) + 1) if (dimension - 1) % (d - 1) == 0]
    if matrix_dim not in valid:
        raise ValueError(
            "NeuroKit error: complexity_lyapunov(): with method='eckmann1986', 'dimension' should be at least 2, "
            + "and 'matrix_dim' should be at most 'dimension', such that 'dimension' - 1 is a multiple of "
            + f"'matrix_dim' - 1, i.e., one of {valid} for dimension={dimension} (matrix_dim={matrix_dim} given)."
        )
    return int(matrix_dim)


def _complexity_lyapunov_checklength(n, separation, n_neighbours=1, details=""):
    """Check that each of the n vectors has n_neighbours vectors more than `separation` samples away."""
    if n < 2 * separation + 1 + n_neighbours:
        raise ValueError(
            "NeuroKit error: complexity_lyapunov(): the signal is too short to find the neighbours of each "
            + "vector that are more than 'separation' samples away in time"
            + (f": {details}" if details else f" ({n} vectors for separation={separation}).")
        )


def _complexity_lyapunov_neighbours(embedded, separation, n_neighbours=1, p=2):
    """Neighbours of each vector, among the vectors that are more than `separation` samples away.

    If n_neighbours is 1, returns the index of the nearest neighbour of each vector (the first one in
    case of ties, as with np.argmin()). Otherwise, returns a list of the indices (sorted) of the
    vectors within the distance of the n_neighbours-th nearest neighbour (ties included). The
    neighbours are found with a KDTree, querying the k nearest vectors (k doubling) until enough of
    them are outside of the `separation` window, which avoids computing the full distance matrix.
    """
    n = len(embedded)
    tree = scipy.spatial.cKDTree(embedded)
    neighbours = np.zeros(n, dtype=int) if n_neighbours == 1 else [None] * n

    todo = np.arange(n)
    k = min(n, n_neighbours + 1)
    while len(todo) > 0:
        k = min(n, 2 * k)
        unresolved = []
        # Query by chunks to limit the memory usage
        for chunk in np.array_split(todo, int(np.ceil(len(todo) * k / 2**20))):
            distance, index = tree.query(embedded[chunk], k=k, p=p)
            valid = np.abs(index - chunk[:, np.newaxis]) > separation
            # Radius: distance of the n_neighbours-th nearest valid neighbour
            radius = np.sort(np.where(valid, distance, np.inf), axis=1)[:, n_neighbours - 1]
            # It is known only if all the vectors at that distance have been returned
            done = (k == n) | (np.isfinite(radius) & (radius < distance[:, -1]))
            unresolved.append(chunk[~done])

            candidates = valid & (distance <= radius[:, np.newaxis])
            if n_neighbours == 1:
                neighbours[chunk[done]] = np.min(np.where(candidates, index, n), axis=1)[done]
            else:
                for row in np.where(done)[0]:
                    neighbours[chunk[row]] = np.sort(index[row][candidates[row]])
        todo = np.concatenate(unresolved)
    return neighbours


def _complexity_lyapunov_localslopes(y, h):
    """Slope of the least-squares line over the steps k - h to k + h (truncated at the edges) of y."""
    x = np.arange(len(y), dtype=float)
    lower = np.maximum(0, np.arange(len(y)) - h)
    upper = np.minimum(len(y) - 1, np.arange(len(y)) + h) + 1
    finite = np.isfinite(y)
    y = np.where(finite, y, 0.0)

    # Sums over the windows, from cumulative sums
    def window_sum(v):
        v = np.concatenate([[0.0], np.cumsum(v)])
        return v[upper] - v[lower]

    count, sx, sy = window_sum(finite), window_sum(x * finite), window_sum(y)
    sxx, sxy = window_sum(x * x * finite), window_sum(x * y)
    with np.errstate(invalid="ignore", divide="ignore"):
        slopes = (sxy - sx * sy / count) / (sxx - sx**2 / count)
    slopes[count < 2] = np.nan
    return slopes


def _complexity_lyapunov_linearrun(slopes, ratio=1.5, share=0.8):
    """First run of positive slopes whose max / min is at most `ratio`, among those at least `share`
    times as long as the longest such run.

    Returns its (start, end) indices (end inclusive), with end < start if no slope is positive.
    """
    # Longest run ending at each index, with the indices of its running minimum and maximum in
    # monotonic queues
    runs = []
    low, high = deque(), deque()
    start = 0
    for end in range(len(slopes)):
        if not slopes[end] > 0:
            low.clear()
            high.clear()
            start = end + 1
            continue
        while low and slopes[low[-1]] >= slopes[end]:
            low.pop()
        low.append(end)
        while high and slopes[high[-1]] <= slopes[end]:
            high.pop()
        high.append(end)
        while slopes[high[0]] > ratio * slopes[low[0]]:
            start += 1
            if low[0] < start:
                low.popleft()
            if high[0] < start:
                high.popleft()
        runs.append((start, end))
    if len(runs) == 0:
        return 0, -1
    longest = max(end - start for start, end in runs)
    return next((start, end) for start, end in runs if end - start >= share * longest)


def _mean_finite(x):
    """Mean of the finite values (NaN if there is none)."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return np.mean(x) if len(x) > 0 else np.nan


def _complexity_lyapunov_plot(le, info, sampling_rate=None):
    trajectories = info["Divergence_Rate"]
    steps = np.arange(len(trajectories))

    # Time axis in the units of the exponent
    scale = 1 if sampling_rate is None else sampling_rate
    time = steps / scale
    intercept = info["Intercept"]

    plt.plot(time, trajectories)
    if info["Method"] == "makowski":
        start, end = info["Fit_Range"]
        plt.axvspan(start / scale, end / scale, color="orange", alpha=0.2, label="Fit range")
        if np.isfinite(info["Plateau"]):
            plt.axhline(info["Plateau"], color="grey", linestyle=":", label="Plateau")
    if np.isfinite(le):
        plt.axline((0, intercept), slope=le, color="orange", label="Least-squares Fit")
    if np.any(np.isfinite(trajectories)):
        plt.ylim(bottom=np.min(trajectories[np.isfinite(trajectories)]))
    plt.xlabel("Time (samples)" if sampling_rate is None else "Time (s)")
    plt.ylabel("Average log divergence")
    plt.title(f"Largest Lyapunov Exponent (slope of the line) = {le:.3f}")
    plt.legend()
