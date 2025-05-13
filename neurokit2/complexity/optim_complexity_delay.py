# -*- coding: utf-8 -*-
import itertools
from warnings import warn

import matplotlib
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import scipy.spatial
import scipy.stats

from ..misc import NeuroKitWarning, find_closest
from ..signal import (
    signal_autocor,
    signal_findpeaks,
    signal_psd,
    signal_surrogate,
    signal_zerocrossings,
)
from .entropy_kl import entropy_kl
from .information_mutual import mutual_information
from .utils_complexity_embedding import complexity_embedding


def complexity_delay(
    signal,
    delay_max=50,
    method="fraser1986",
    algorithm=None,
    show=False,
    silent=False,
    **kwargs
):
    """**Automated selection of the optimal Delay (Tau)**

    The time delay (Tau :math:`\\tau`, also referred to as *Lag*) is one of the two critical
    parameters (the other being the :func:`Dimension <complexity_dimension>` *m*) involved in the
    construction of the time-delay embedding of a signal. It corresponds to the delay in samples
    between the original signal and its delayed version(s). In other words, how many samples do we
    consider between a given state of the signal and its closest past state.

    When :math:`\\tau` is smaller than the optimal theoretical value, consecutive coordinates of the
    system's state are correlated and the attractor is not sufficiently unfolded. Conversely, when
    :math:`\\tau` is larger than it should be, successive coordinates are almost independent,
    resulting in an uncorrelated and unstructured cloud of points.

    The selection of the parameters *delay* and :func:`*dimension* <complexity_dimension>` is a
    challenge. One approach is to select them (semi) independently (as dimension selection often
    requires the delay), using :func:`complexity_delay` and :func:`complexity_dimension`. However,
    some joint-estimation methods do exist, that attempt at finding the optimal delay and dimension
    at the same time.

    Note also that some authors (e.g., Rosenstein, 1994) suggest identifying the
    optimal embedding dimension first, and that the optimal delay value should then be considered
    as the optimal delay between the first and last delay coordinates (in other words, the actual
    delay should be the optimal delay divided by the optimal embedding dimension minus 1).

    Several authors suggested different methods to guide the choice of the delay:

    * **Fraser and Swinney (1986)** suggest using the first local minimum of the mutual information
      between the delayed and non-delayed time series, effectively identifying a value of Tau for
      which they share the least information (and where the attractor is the least redundant).
      Unlike autocorrelation, mutual information takes into account also nonlinear correlations.
    * **Theiler (1990)** suggested to select Tau where the autocorrelation between the signal and
      its lagged version at Tau first crosses the value :math:`1/e`. The autocorrelation-based
      methods have the advantage of short computation times when calculated via the fast Fourier
      transform (FFT) algorithm.
    * **Casdagli (1991)** suggests instead taking the first zero-crossing of the autocorrelation.
    * **Rosenstein (1993)** suggests to approximate the point where the autocorrelation function
      drops to :math:`(1 - 1/e)` of its maximum value.
    * **Rosenstein (1994)** suggests to the point close to 40% of the slope of the average
      displacement from the diagonal (ADFD).
    * **Kim (1999)** suggests estimating Tau using the correlation integral, called the C-C method,
      which has shown to agree with those obtained using the Mutual Information. This method
      makes use of a statistic within the reconstructed phase space, rather than analyzing the
      temporal evolution of the time series. However, computation times are significantly long for
      this method due to the need to compare every unique pair of pairwise vectors within the
      embedded signal per delay.
    * **Lyle (2021)** describes the "Symmetric Projection Attractor Reconstruction" (SPAR), where
      :math:`1/3` of the the dominant frequency (i.e., of the length of the average "cycle") can be
      a suitable value for approximately periodic data, and makes the attractor sensitive to
      morphological changes. See also `Aston's talk <https://youtu.be/GGrOJtcTcHA?t=730>`_. This
      method is also the fastest but might not be suitable for aperiodic signals.
      The ``algorithm`` argument (default to ``"fft"``) and will be passed as the ``method``
      argument of  ``signal_psd()``.

    **Joint-Methods for Delay and Dimension**

    * **Gautama (2003)** mentions that in practice, it is common to have a fixed time lag and to
      adjust the embedding dimension accordingly. As this can lead to large *m* values (and thus to
      embedded data of a large size) and thus, slow processing, they describe an optimisation
      method to jointly determine *m* and :math:`\\tau`, based on the **entropy ratio**.

    .. note::

      We would like to implement the joint-estimation by `Matilla-García et al. (2021)
      <https://www.mdpi.com/1099-4300/23/2/221>`_, please get in touch if you can help us!

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay_max : int
        The maximum time delay (Tau or lag) to test.
    method : str
        The method that defines what to compute for each tested value of Tau. Can be one of
        ``"fraser1986"``, ``"theiler1990"``, ``"casdagli1991"``, ``"rosenstein1993"``,
        ``"rosenstein1994"``, ``"kim1999"``, or ``"lyle2021"``.
    algorithm : str
        The method used to find the optimal value of Tau given the values computed by the method.
        If ``None`` (default), will select the algorithm according to the method. Modify only if you
        know what you are doing.
    show : bool
        If ``True``, will plot the metric values for each value of tau.
    silent : bool
        If ``True``, silence possible warnings.
    **kwargs : optional
        Additional arguments to be passed for C-C method.

    Returns
    -------
    delay : int
        Optimal time delay.
    parameters : dict
        A dictionary containing additional information regarding the parameters used
        to compute optimal time-delay embedding.

    See Also
    ---------
    complexity, complexity_dimension, complexity_embedding, complexity_tolerance

    Examples
    ----------

    * **Example 1**: Comparison of different methods for estimating the optimal delay of an simple
      artificial signal.

    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=10, sampling_rate=100, frequency=[1, 1.5], noise=0.02)

      @savefig p_complexity_delay1.png scale=100%
      nk.signal_plot(signal)
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_delay2.png scale=100%
      delay, parameters = nk.complexity_delay(signal, delay_max=100, show=True,
                                              method="fraser1986")
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_delay3.png scale=100%
      delay, parameters = nk.complexity_delay(signal, delay_max=100, show=True,
                                              method="theiler1990")
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_delay4.png scale=100%
      delay, parameters = nk.complexity_delay(signal, delay_max=100, show=True,
                                              method="casdagli1991")
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_delay5.png scale=100%
      delay, parameters = nk.complexity_delay(signal, delay_max=100, show=True,
                                              method="rosenstein1993")
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_delay6.png scale=100%
      delay, parameters = nk.complexity_delay(signal, delay_max=100, show=True,
                                              method="rosenstein1994")
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_delay7.png scale=100%
      delay, parameters = nk.complexity_delay(signal, delay_max=100, show=True,
                                              method="lyle2021")
      @suppress
      plt.close()

    * **Example 2**: Optimizing the delay and the dimension using joint-estimation methods.

    .. ipython:: python

      @savefig p_complexity_delay8.png scale=100%
      delay, parameters = nk.complexity_delay(
          signal,
          delay_max=np.arange(1, 30, 1),  # Can be an int or a list
          dimension_max=20,  # Can be an int or a list
          method="gautama2003",
          surrogate_n=5,  # Number of surrogate signals to generate
          surrogate_method="random",  # can be IAAFT, see nk.signal_surrogate()
          show=True)
      @suppress
      plt.close()

    .. ipython:: python

      # Optimal dimension
      dimension = parameters["Dimension"]
      dimension

    **Note**: A double-checking of that method would be appreciated! Please help us improve.

    * **Example 3**: Using a realistic signal.

    .. ipython:: python

      ecg = nk.ecg_simulate(duration=60*6, sampling_rate=200)
      signal = nk.ecg_rate(nk.ecg_peaks(ecg, sampling_rate=200),
                           sampling_rate=200,
                           desired_length=len(ecg))
      @savefig p_complexity_delay9.png scale=100%
      nk.signal_plot(signal)
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_delay10.png scale=100%
      delay, parameters = nk.complexity_delay(signal, delay_max=1000, show=True)
      @suppress
      plt.close()

    References
    ------------
    * Lyle, J. V., Nandi, M., & Aston, P. J. (2021). Symmetric Projection Attractor Reconstruction:
      Sex Differences in the ECG. Frontiers in cardiovascular medicine, 1034.
    * Gautama, T., Mandic, D. P., & Van Hulle, M. M. (2003, April). A differential entropy based
      method for determining the optimal embedding parameters of a signal. In 2003 IEEE
      International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings.
      (ICASSP'03). (Vol. 6, pp. VI-29). IEEE.
    * Camplani, M., & Cannas, B. (2009). The role of the embedding dimension and time delay in time
      series forecasting. IFAC Proceedings Volumes, 42(7), 316-320.
    * Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993). A practical method for calculating
      largest Lyapunov exponents from small data sets. Physica D: Nonlinear Phenomena, 65(1-2),
      117-134.
    * Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1994). Reconstruction expansion as a
      geometry-based framework for choosing proper delay times. Physica-Section D, 73(1), 82-98.
    * Kim, H., Eykholt, R., & Salas, J. D. (1999). Nonlinear dynamics, delay times, and embedding
      windows. Physica D: Nonlinear Phenomena, 127(1-2), 48-60.
    * Gautama, T., Mandic, D. P., & Van Hulle, M. M. (2003, April). A differential entropy based
      method for determining the optimal embedding parameters of a signal. In 2003 IEEE
      International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings.
      (ICASSP'03). (Vol. 6, pp. VI-29). IEEE.
    * Camplani, M., & Cannas, B. (2009). The role of the embedding dimension and time delay in time
      series forecasting. IFAC Proceedings Volumes, 42(7), 316-320.

    """
    # Initalize vectors
    if isinstance(delay_max, int):
        tau_sequence = np.arange(1, delay_max + 1)
    else:
        tau_sequence = np.array(delay_max).astype(int)

    # Method
    method = method.lower()
    if method in ["fraser", "fraser1986", "tdmi"]:
        metric = "Mutual Information"
        if algorithm is None:
            algorithm = "first local minimum"
    elif method in ["mi2"]:
        metric = "Mutual Information 2"
        if algorithm is None:
            algorithm = "first local minimum"
    elif method in ["theiler", "theiler1990"]:
        metric = "Autocorrelation"
        if algorithm is None:
            algorithm = "first 1/e crossing"
    elif method in ["casdagli", "casdagli1991"]:
        metric = "Autocorrelation"
        if algorithm is None:
            algorithm = "first zero crossing"
    elif method in ["rosenstein", "rosenstein1994", "adfd"]:
        metric = "Displacement"
        if algorithm is None:
            algorithm = "closest to 40% of the slope"
    elif method in ["rosenstein1993"]:
        metric = "Autocorrelation"
        if algorithm is None:
            algorithm = "first drop below 1-(1/e) of maximum"
    elif method in ["kim1999", "cc"]:
        metric = "Correlation Integral"
        if algorithm is None:
            algorithm = "first local minimum"
    elif method in ["aston2020", "lyle2021", "spar"]:
        return _complexity_delay_spar(signal, algorithm=algorithm, show=show, **kwargs)
    elif method in ["gautama2003", "gautama", "entropyratio"]:
        return _complexity_delay_entropyratio(
            signal, delay_max=delay_max, show=show, **kwargs
        )
    else:
        raise ValueError("complexity_delay(): 'method' not recognized.")

    # Get metric
    metric_values = _embedding_delay_metric(signal, tau_sequence, metric=metric)

    # Get optimal tau
    optimal = _embedding_delay_select(metric_values, algorithm=algorithm)

    if np.isnan(optimal) and silent is False:
        warn(
            "No optimal time delay is found. Nan is returned."
            " Consider using a higher `delay_max`.",
            category=NeuroKitWarning,
        )
    else:
        optimal = tau_sequence[optimal]

        if show is True:
            _embedding_delay_plot(
                signal,
                metric_values=metric_values,
                tau_sequence=tau_sequence,
                tau=optimal,
                metric=metric,
            )

    # Return optimal tau and info dict
    return optimal, {
        "Values": tau_sequence,
        "Scores": metric_values,
        "Algorithm": algorithm,
        "Metric": metric,
        "Method": method,
    }


# =============================================================================
# Methods
# =============================================================================
def _embedding_delay_select(metric_values, algorithm="first local minimum"):

    if algorithm in ["first local minimum (corrected)"]:
        # if immediately increasing, assume it is the first is the closest
        if np.diff(metric_values)[0] > 0:
            optimal = 0
        # take last value if continuously decreasing with no inflections
        elif all(np.diff(metric_values) < 0):
            optimal = len(metric_values) - 1
        else:
            # Find reversed peaks
            optimal = signal_findpeaks(
                -1 * metric_values, relative_height_min=0.1, relative_max=True
            )["Peaks"]
    elif algorithm == "first local minimum":
        # Find reversed peaks
        try:
            optimal = signal_findpeaks(
                -1 * metric_values, relative_height_min=0.1, relative_max=True
            )["Peaks"]
        except ValueError:
            warn(
                "First local minimum detection failed. Try setting "
                + "`algorithm = 'first local minimum (corrected)'` or using another method.",
                category=NeuroKitWarning,
            )
            optimal = np.nan

    elif algorithm == "first 1/e crossing":
        metric_values = metric_values - 1 / np.exp(1)
        optimal = signal_zerocrossings(metric_values)
    elif algorithm == "first zero crossing":
        optimal = signal_zerocrossings(metric_values)
    elif algorithm == "closest to 40% of the slope":
        slope = np.diff(metric_values) * len(metric_values)
        slope_in_deg = np.rad2deg(np.arctan(slope))
        optimal = np.where(slope_in_deg == find_closest(40, slope_in_deg))[0]
    elif algorithm == "first drop below 1-(1/e) of maximum":
        try:
            optimal = np.where(
                metric_values < np.nanmax(metric_values) * (1 - 1.0 / np.e)
            )[0][0]
        except IndexError:  # If no value are below that threshold
            optimal = np.nan

    if not isinstance(optimal, (int, float, np.integer)):
        if len(optimal) != 0:
            optimal = optimal[0]
        else:
            optimal = np.nan

    return optimal


# =============================================================================
def _embedding_delay_metric(
    signal,
    tau_sequence,
    metric="Mutual Information",
    dimensions=[2, 3, 4, 5],
    r_vals=[0.5, 1.0, 1.5, 2.0],
):
    """Iterating through dimensions and r values is relevant only if metric used is Correlation Integral.
    For this method, either first zero crossing of the statistic averages or the first local
    minimum of deviations to obtain optimal tau. This implementation takes the latter since in practice,
    they are both in close proximity.
    """

    if metric == "Autocorrelation":
        values, _ = signal_autocor(signal)
        values = values[: len(tau_sequence)]  # upper limit

    elif metric == "Correlation Integral":
        r_vals = [i * np.std(signal) for i in r_vals]
        # initiate empty list for storing
        # averages = np.zeros(len(tau_sequence))
        values = np.zeros(len(tau_sequence))
        for i, t in enumerate(tau_sequence):
            # average = 0
            change = 0
            for m in dimensions:
                # # find average of dependence statistic
                # for r in r_vals:
                #     s = _embedding_delay_cc_statistic(signal, delay=t, dimension=m, r=r)
                #     average += s

                # find average of statistic deviations across r_vals
                deviation = _embedding_delay_cc_deviation_max(
                    signal, delay=t, dimension=m, r_vals=r_vals
                )
                change += deviation
            # averages[i] = average / 16
            values[i] = change / 4

    else:
        values = np.zeros(len(tau_sequence))

        # Loop through taus and compute all scores values
        for i, current_tau in enumerate(tau_sequence):
            embedded = complexity_embedding(signal, delay=current_tau, dimension=2)
            if metric == "Mutual Information":
                values[i] = mutual_information(
                    embedded[:, 0], embedded[:, 1], method="varoquaux"
                )
            elif metric == "Mutual Information 2":
                values[i] = mutual_information(
                    embedded[:, 0], embedded[:, 1], method="knn"
                )
            elif metric == "Displacement":
                dimension = 2

                # Reconstruct with zero time delay.
                tau0 = (
                    embedded[:, 0].repeat(dimension).reshape(len(embedded), dimension)
                )
                dist = np.asarray(
                    [
                        scipy.spatial.distance.euclidean(i, j)
                        for i, j in zip(embedded, tau0)
                    ]
                )
                values[i] = np.mean(dist)
            else:
                raise ValueError("'metric' not recognized.")

    return values


# =============================================================================
# Internals for C-C method, Kim et al. (1999)
# =============================================================================


def _embedding_delay_cc_integral_sum(signal, dimension=3, delay=10, r=0.02):
    """Correlation integral is a cumulative distribution function, which denotes
    the probability of distance between any pairs of points in phase space
    not greater than the specified `r`.
    """

    # Embed signal
    embedded = complexity_embedding(
        signal, delay=delay, dimension=dimension, show=False
    )
    M = embedded.shape[0]

    # Prepare indices for comparing all unique pairwise vectors
    combinations = list(itertools.combinations(range(0, M), r=2))
    first_index, second_index = (
        np.transpose(combinations)[0],
        np.transpose(combinations)[1],
    )

    vectorized_integral = np.vectorize(
        _embedding_delay_cc_integral, excluded=["embedded", "r"]
    )
    integral = np.sum(
        vectorized_integral(
            first_index=first_index, second_index=second_index, embedded=embedded, r=r
        )
    )

    return integral


def _embedding_delay_cc_integral(first_index, second_index, embedded, r=0.02):
    M = embedded.shape[0]  # Number of embedded points
    diff = np.linalg.norm(
        embedded[first_index] - embedded[second_index], ord=np.inf
    )  # sup-norm
    h = np.heaviside(r - diff, 1)
    integral = (2 / (M * (M - 1))) * h  # find average

    return integral


def _embedding_delay_cc_statistic(signal, dimension=3, delay=10, r=0.02):
    """The dependence statistic as the serial correlation of a nonlinear time series."""

    # create disjoint time series
    series = [signal[i - 1 :: delay] for i in range(1, delay + 1)]

    statistic = 0
    for sub_series in series:
        diff = _embedding_delay_cc_integral_sum(
            sub_series, dimension=dimension, delay=delay, r=r
        ) - (
            (_embedding_delay_cc_integral_sum(signal, dimension=1, delay=delay, r=r))
            ** dimension
        )
        statistic += diff

    return statistic / delay


def _embedding_delay_cc_deviation_max(
    signal, r_vals=[0.5, 1.0, 1.5, 2.0], delay=10, dimension=3
):
    """A measure of the variation of the dependence statistic with r using
    several representative values of r.
    """
    vectorized_deviation = np.vectorize(
        _embedding_delay_cc_deviation, excluded=["signal", "delay", "dimension"]
    )
    deviations = vectorized_deviation(
        signal=signal, r_vals=r_vals, delay=delay, dimension=dimension
    )

    return np.max(deviations) - np.min(deviations)


def _embedding_delay_cc_deviation(
    signal, r_vals=[0.5, 1.0, 1.5, 2.0], delay=10, dimension=3
):
    return _embedding_delay_cc_statistic(
        signal, delay=delay, dimension=dimension, r=r_vals
    )


# =============================================================================
# Plotting Generics
# =============================================================================
def _embedding_delay_plot(
    signal,
    metric_values,
    tau_sequence,
    tau=1,
    metric="Mutual Information",
    ax0=None,
    ax1=None,
    plot="2D",
):

    # Prepare figure
    if ax0 is None and ax1 is None:
        fig = plt.figure(constrained_layout=False)
        spec = matplotlib.gridspec.GridSpec(
            ncols=1, nrows=2, height_ratios=[1, 3], width_ratios=[2]
        )
        ax0 = fig.add_subplot(spec[0])
        if plot == "2D":
            ax1 = fig.add_subplot(spec[1])
        elif plot == "3D":
            ax1 = fig.add_subplot(spec[1], projection="3d")
    else:
        fig = None

    ax0.set_title("Optimization of Delay")
    ax0.set_xlabel("Time Delay")
    ax0.set_ylabel(metric)
    ax0.plot(tau_sequence, metric_values, color="#FFC107")
    ax0.axvline(x=tau, color="#E91E63", label="Optimal delay: " + str(tau))
    ax0.legend(loc="upper right")
    ax1.set_title("Attractor")
    ax1.set_xlabel("Signal [i]")
    ax1.set_ylabel("Signal [i-" + str(tau) + "]")

    # Get data points, set axis limits
    embedded = complexity_embedding(signal, delay=tau, dimension=3)
    x = embedded[:, 0]
    y = embedded[:, 1]
    z = embedded[:, 2]
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(x.min(), x.max())

    # Colors
    norm = plt.Normalize(z.min(), z.max())
    cmap = plt.get_cmap("plasma")
    colors = cmap(norm(x))

    # Attractor for 2D vs 3D
    if plot == "2D":
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = matplotlib.collections.LineCollection(segments, cmap="plasma", norm=norm)
        lc.set_array(z)
        ax1.add_collection(lc)

    elif plot == "3D":
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        for i in range(len(x) - 1):
            seg = segments[i]
            (l,) = ax1.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=colors[i])
            l.set_solid_capstyle("round")
        ax1.set_zlabel("Signal [i-" + str(2 * tau) + "]")

    return fig


# =============================================================================
# Optimal Delay via SPAR Method
# =============================================================================


def _complexity_delay_spar(signal, algorithm=None, show=False, **kwargs):
    if algorithm is None:
        algorithm = "fft"
    # Compute power in freqency domain
    psd = signal_psd(signal, sampling_rate=1000, method=algorithm, show=False, **kwargs)
    power = psd["Power"].values
    freqs = 1000 / psd["Frequency"].values  # Convert to samples

    # Get the 1/3 max frequency (in samples) (https://youtu.be/GGrOJtcTcHA?t=730)
    idx = np.argmax(power)
    optimal = int(freqs[idx] / 3)

    if show is True:
        idxs = freqs <= optimal * 6
        _embedding_delay_plot(
            signal,
            metric_values=power[idxs],
            tau_sequence=freqs[idxs],
            tau=optimal,
            metric="Power",
        )
    return optimal, {"Algorithm": algorithm, "Method": "SPAR"}


# =============================================================================
# Joint-Optimization via Entropy Ratio (Gautama, 2003)
# =============================================================================


def _complexity_delay_entropyratio(
    signal,
    delay_max=20,
    dimension_max=5,
    surrogate_n=5,
    surrogate_method="random",
    show=False,
):
    """Joint-optimization using Entropy Ratio method.

    * Gautama, T., Mandic, D. P., & Van Hulle, M. M. (2003, April). A differential entropy based
      method for determining the optimal embedding parameters of a signal. In 2003 IEEE
      International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings.
      (ICASSP'03). (Vol. 6, pp. VI-29). IEEE.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay_max : int
        The maximum time delay (Tau) to test.
    dimension_max : int
        The maximum embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order') to test.
    surrogate_iter : int
        The maximum surrogates generated using the iAAFT method.

    """

    # Initalize vectors
    if isinstance(delay_max, int):
        delay_max = np.arange(1, delay_max + 1)
    if isinstance(dimension_max, int):
        dimension_max = np.arange(2, dimension_max + 1)

    # Prepare output
    out = pd.DataFrame(
        {
            "Dimension": np.repeat(dimension_max, len(delay_max)),
            "Delay": np.tile(delay_max, len(dimension_max)),
        }
    )
    # N is the number of delay vectors
    # out["N"] = len(signal) - (out["Dimension"] - 1) * out["Delay"]
    out["N"] = len(signal)

    klens = np.full(len(out), np.nan)
    klens_surrogates = np.full(len(out), np.nan)

    for i in range(len(out)):
        klens[i], _ = entropy_kl(
            signal,
            delay=out["Delay"][i],
            dimension=out["Dimension"][i],
        )
        klens_surrogates[i] = np.nanmean(
            [
                entropy_kl(
                    signal_surrogate(signal, method=surrogate_method),
                    delay=out["Delay"][i],
                    dimension=out["Dimension"][i],
                )[0]
                for j in range(surrogate_n)
            ]
        )

    out["KLEn_Signal"] = klens
    out["KLEn_Surrogate"] = klens_surrogates
    # Entropy Ratio (ER)
    out["KLEn_Ratio"] = klens / klens_surrogates
    # To penalise for higher embedding dimensions, the minimum
    # description length (MDL) method is superimposed, yielding
    # the "entropy ratio" (ER):
    out["Entropy_Ratio"] = out["KLEn_Ratio"] * (
        1 + (out["Dimension"] * np.log(out["N"])) / out["N"]
    )

    # optimal dimension and tau is where entropy_ratio is minimum
    idx = out["Entropy_Ratio"].argmin()
    optimal_dimension = out["Dimension"][idx]
    optimal_delay = out["Delay"][idx]

    if show is True:
        plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_title("Joint-Estimation of Optimal Delay and Dimension")
        ax.plot_trisurf(
            out["Delay"],
            out["Dimension"],
            out["Entropy_Ratio"],
            cmap=plt.get_cmap("GnBu"),
            antialiased=True,
            linewidth=0,
            label="Minimum Entropy Ratio",
            zorder=1,
        )
        ax.plot(
            [optimal_delay] * 2,
            [optimal_dimension] * 2,
            [out["Entropy_Ratio"].min(), out["Entropy_Ratio"].max()],
            zorder=2,
        )
        ax.scatter(
            [optimal_delay] * 2,
            [optimal_dimension] * 2,
            [out["Entropy_Ratio"].min(), out["Entropy_Ratio"].max()],
            color="red",
            zorder=2,
        )
        ax.set_ylabel("Dimension")
        ax.set_xlabel("Delay")
        ax.set_zlabel("Entropy Ratio")
        if len(dimension_max) < 10:
            ax.set_yticks(dimension_max)
        if len(delay_max) < 10:
            ax.set_xticks(delay_max)

    return optimal_delay, {"Dimension": optimal_dimension, "Data": out}


# =============================================================================
# Joint-Optimization via SymbolicDynamic (Matilla-García, 2021)
# =============================================================================


# def _complexity_delay_symbolicdynamics(
#     signal,
#     delay_max=20,
#     dimension_max=5,
#     show=False,
# ):
#     """https://www.mdpi.com/1099-4300/23/2/221/htm"""

#     delay = 1
#     dimension = 3
#     e = 3

#     signal = [2, -7, -12, 5, -1, 9, 14]

#     embedded = nk.complexity_embedding(signal, delay=delay, dimension=dimension)
#     print(embedded)

#     # How to create the symbolic sequence?
#     symbols = np.zeros((len(embedded), dimension - 1))
#     for d in range(1, dimension):
#         # Difference with e
#         symbols[:, d - 1][np.all(np.abs(embedded[:, d - 1 : d + 1]) >= e, axis=1)] = 1

#     symbols[:, 0][np.abs(embedded[:, 0]) >= e] = 1
#     symbols[:, 1][np.all(embedded[:, 1:] < e, axis=1)] = 1
