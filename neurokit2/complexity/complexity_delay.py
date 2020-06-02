# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import scipy.spatial
import scipy.stats

from ..misc import find_closest
from ..signal import signal_autocor, signal_findpeaks, signal_zerocrossings
from ..stats import mutual_information
from .complexity_embedding import complexity_embedding


def complexity_delay(signal, delay_max=100, method="fraser1986", show=False):
    """
    Estimate optimal Time Delay (tau) for time-delay embedding.

    The time delay (Tau) is one of the two critical parameters involved in the construction of the time-delay embedding of a signal.

    Several authors suggested different methods to guide the choice of Tau. Fraser and Swinney (1986) suggest using the first local minimum of the mutual information between the delayed and non-delayed time series, effectively identifying a value of tau for which they share the least information. Theiler (1990) suggested to select Tau where the autocorrelation between the signal and its lagged version at Tau first crosses the value 1/e. Casdagli (1991) suggests instead taking the first zero-crossing of the autocorrelation. Rosenstein (1993) suggests to the point close to 40% of the slope of the average displacement from the diagonal (ADFD).

    Parameters
    ----------
    signal : list, array or Series
        The signal (i.e., a time series) in the form of a vector of values.
    delay_max : int
        The maximum time delay (Tau or lag) to test.
    method : str
        Correlation method. Can be one of 'fraser1986', 'theiler1990', 'casdagli1991', 'rosenstein1993'.
    show : bool
        If true, will plot the mutual information values for each value of tau.

    Returns
    -------
    int
        Optimal time delay.

    See Also
    ---------
    complexity_dimension, complexity_embedding

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Artifical example
    >>> signal = nk.signal_simulate(duration=10, frequency=1, noise=0.01)
    >>> nk.signal_plot(signal)
    >>>
    >>> delay = nk.complexity_delay(signal, delay_max=1000, show=True, method="fraser1986")
    >>> delay = nk.complexity_delay(signal, delay_max=1000, show=True, method="theiler1990")
    >>> delay = nk.complexity_delay(signal, delay_max=1000, show=True, method="casdagli1991")
    >>> delay = nk.complexity_delay(signal, delay_max=1000, show=True, method="rosenstein1993")
    >>>
    >>> # Realistic example
    >>> ecg = nk.ecg_simulate(duration=60*6, sampling_rate=150)
    >>> signal = nk.ecg_rate(nk.ecg_peaks(ecg, sampling_rate=150), sampling_rate=150)
    >>> nk.signal_plot(signal)
    >>>
    >>> delay = nk.complexity_delay(signal, delay_max=1000, show=True)

    References
    ------------
    - Gautama, T., Mandic, D. P., & Van Hulle, M. M. (2003, April). A differential entropy based method for determining the optimal embedding parameters of a signal. In 2003 IEEE International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings.(ICASSP'03). (Vol. 6, pp. VI-29). IEEE.
    - Camplani, M., & Cannas, B. (2009). The role of the embedding dimension and time delay in time series forecasting. IFAC Proceedings Volumes, 42(7), 316-320.
    - Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1994). Reconstruction expansion as a geometry-based framework for choosing proper delay times. Physica-Section D, 73(1), 82-98.

    """
    # Initalize vectors
    if isinstance(delay_max, int):
        tau_sequence = np.arange(1, delay_max)
    else:
        tau_sequence = np.array(delay_max)

    # Method
    method = method.lower()
    if method in ["fraser", "fraser1986", "tdmi"]:
        metric = "Mutual Information"
        algorithm = "first local minimum"
    elif method in ["theiler", "theiler1990"]:
        metric = "Autocorrelation"
        algorithm = "first 1/e crossing"
    elif method in ["casdagli", "casdagli1991"]:
        metric = "Autocorrelation"
        algorithm = "first zero crossing"
    elif method in ["rosenstein", "rosenstein1993", "adfd"]:
        metric = "Displacement"
        algorithm = "closest to 40% of the slope"
    else:
        raise ValueError("NeuroKit error: complexity_delay(): 'method' not recognized.")

    # Get metric
    metric_values = _embedding_delay_metric(signal, tau_sequence, metric=metric)

    # Get optimal tau
    optimal = _embedding_delay_select(metric_values, algorithm=algorithm)
    optimal = tau_sequence[optimal]

    if show is True:
        _embedding_delay_plot(
            signal, metric_values=metric_values, tau_sequence=tau_sequence, tau=optimal, metric=metric
        )

    return optimal


# =============================================================================
# Methods
# =============================================================================
def _embedding_delay_select(metric_values, algorithm="first local minimum"):

    if algorithm == "first local minimum":
        optimal = signal_findpeaks(-1 * metric_values, relative_height_min=0.1, relative_max=True)["Peaks"][0]
    elif algorithm == "first 1/e crossing":
        metric_values = metric_values - 1 / np.exp(1)
        optimal = signal_zerocrossings(metric_values)[0]
    elif algorithm == "first zero crossing":
        optimal = signal_zerocrossings(metric_values)[0]
    elif algorithm == "closest to 40% of the slope":
        slope = np.diff(metric_values) * len(metric_values)
        slope_in_deg = np.rad2deg(np.arctan(slope))
        optimal = np.where(slope_in_deg == find_closest(40, slope_in_deg))[0][0]
    return optimal


def _embedding_delay_metric(signal, tau_sequence, metric="Mutual Information"):

    if metric == "Autocorrelation":
        values = signal_autocor(signal)
        values = values[: len(tau_sequence)]  # upper limit

    else:
        values = np.zeros(len(tau_sequence))

        # Loop through taus and compute all scores values
        for i, current_tau in enumerate(tau_sequence):
            embedded = complexity_embedding(signal, delay=current_tau, dimension=2)
            if metric == "Mutual Information":
                values[i] = mutual_information(embedded[:, 0], embedded[:, 1], normalized=True, method="shannon")
            if metric == "Displacement":
                dimension = 2

                # Reconstruct with zero time delay.
                tau0 = embedded[:, 0].repeat(dimension).reshape(len(embedded), dimension)
                dist = np.asarray([scipy.spatial.distance.euclidean(i, j) for i, j in zip(embedded, tau0)])
                values[i] = np.mean(dist)

    return values


# =============================================================================
# Internals
# =============================================================================
def _embedding_delay_plot(
    signal, metric_values, tau_sequence, tau=1, metric="Mutual Information", ax0=None, ax1=None, plot="2D"
):

    # Prepare figure
    if ax0 is None and ax1 is None:
        fig = plt.figure(constrained_layout=False)
        spec = matplotlib.gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 3], width_ratios=[2])
        ax0 = fig.add_subplot(spec[0])
        if plot == "2D":
            ax1 = fig.add_subplot(spec[1])
        elif plot == "3D":
            ax1 = fig.add_subplot(spec[1], projection="3d")
    else:
        fig = None

    ax0.set_title("Optimization of Delay (tau)")
    ax0.set_xlabel("Time Delay (tau)")
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
        line = ax1.add_collection(lc)

    elif plot == "3D":
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        for i in range(len(x) - 1):
            seg = segments[i]
            (l,) = ax1.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=colors[i])
            l.set_solid_capstyle("round")
        ax1.set_zlabel("Signal [i-" + str(2 * tau) + "]")

    return fig
