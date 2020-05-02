# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.signal
import matplotlib
import matplotlib.collections
import matplotlib.pyplot as plt

from ..stats import cor
from ..misc import findclosest
from ..signal import signal_findpeaks
from .embedding import embedding
from .mutual_information import mutual_information


def embedding_delay(signal, delay_max=100, method="fraser1986", show=False):
    """Estimate optimal Time Delay (tau) for Time-delay embedding

    The time delay (Tau) is one of the two critical parameters involved in the construction of the time-delay embedding of a signal.

    Several authors suggested different methods to guide the choice of Tau. Fraser and Swinney (1986) suggest using the first local minimum of the mutual information between the delayed and non-delayed time series, effectively identifying a value of tau for which they share the least information. Theiler (1990) suggested to select Tau such that the autocorrelation between the signal and its lagged version at Tau is the closest to 1/e. Casdagli (1991) suggests instead taking the first zero-crossing of the autocorrelation.

    The code is based on http://node99.org/tutorials/ar/, but very unsure of our implementation.
    Please help us by checking-it.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    delay_max : int
        The maximum time delay (Tau) to test.
    method : str
        Correlation method. Can be one of 'fraser1986', 'theiler1990', 'casdagli1991'.
    show : bool
        If true, will plot the mutual information values for each value of tau.

    Returns
    -------
    int
        Optimal time delay.

    See Also
    ---------
    embedding

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Artifical example
    >>> signal = nk.signal_simulate(duration=10, frequency=1, noise=0.01)
    >>> nk.signal_plot(signal)
    >>>
    >>> tau = nk.embedding_delay(signal, tau_max=1000, show=True, method="fraser1986")
    >>> tau = nk.embedding_delay(signal, tau_max=1000, show=True, method="theiler1990")
    >>> tau = nk.embedding_delay(signal, tau_max=1000, show=True, method="casdagli1991")
    >>>
    >>> # Realistic example
    >>> ecg = nk.ecg_simulate(duration=120, sampling_rate=200)
    >>> signal = nk.ecg_rate(nk.ecg_peaks(ecg, sampling_rate=200)[0], sampling_rate=200)
    >>> nk.signal_plot(signal)
    >>>
    >>> tau = nk.embedding_delay(signal, tau_max=1000, show=True)

    References
    ------------
    - Gautama, T., Mandic, D. P., & Van Hulle, M. M. (2003, April). A differential entropy based method for determining the optimal embedding parameters of a signal. In 2003 IEEE International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings.(ICASSP'03). (Vol. 6, pp. VI-29). IEEE.
    - Camplani, M., & Cannas, B. (2009). The role of the embedding dimension and time delay in time series forecasting. IFAC Proceedings Volumes, 42(7), 316-320.
    """
    # Initalize vectors
    if isinstance(delay_max, int):
        tau_sequence = np.arange(1, delay_max)
    else:
        tau_sequence = np.array(delay_max)

    # Method
    method = method.lower()
    if method in ["fraser", "fraser1986"]:
        metric = "Mutual Information"
        algorithm = "first local minimum"
    elif method in ["theiler", "theiler1990"]:
        metric = "Autocorrelation"
    elif method in ["casdagli", "casdagli1991 "]:
        metric = "Autocorrelation"
        algorithm = "closest to 0"
    else:
        raise ValueError("NeuroKit error: embedding_delay(): 'method' "
                         "not recognized.")

    # Get metric
    metric_values = _embedding_delay_metric(signal, tau_sequence, metric=metric)

    # Get optimal tau
    optimal = _embedding_delay_select(metric_values, algorithm=algorithm)
    optimal = tau_sequence[optimal]

    if show is True:
        _optimal_delay_plot(signal,
                            metric_values=metric_values,
                            tau_sequence=tau_sequence,
                            tau=optimal,
                            metric=metric)

    return optimal




# =============================================================================
# Methods
# =============================================================================
def _embedding_delay_select(metric_values, algorithm="first local minimum"):

    if algorithm == "first local minimum":
        optimal = signal_findpeaks(-1 * metric_values, relative_height_min=0.1, relative_max=True)["Peaks"][0]
    elif algorithm == "closest to 1/e":
        optimal = np.where(metric_values == findclosest(1 / np.exp(1), metric_values))[0][0]
    elif algorithm == "closest to 0":
        optimal = np.where(metric_values == findclosest(0, metric_values))[0][0]
    return optimal



def _embedding_delay_metric(signal, tau_sequence, metric="Mutual Information"):

    values = np.zeros(len(tau_sequence))

    # Loop through taus and compute all scores values
    for i, current_tau in enumerate(tau_sequence):
        embedded = embedding(signal, delay=current_tau, dimension=2)
        if metric == "Mutual Information":
            values[i] = mutual_information(embedded[:, 0], embedded[:, 1], normalized=True)
        if metric == "Autocorrelation":
            values[i] = cor(embedded[:, 0], embedded[:, 1])
    return values


# =============================================================================
# Internals
# =============================================================================
def _optimal_delay_plot(signal, metric_values, tau_sequence, tau=1, metric="Mutual Information"):
    """
    """
    fig = plt.figure(constrained_layout=False)
    spec = matplotlib.gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 3], width_ratios=[2])

    # Upper plot (metric evolution)
    ax0 = fig.add_subplot(spec[0])
    ax0.set_xlabel("Time Delay (tau)")
    ax0.set_ylabel(metric)
    ax0.plot(tau_sequence, metric_values, color='#2196F3')
    ax0.axvline(x=tau, color='#E91E63', label='Optimal delay: ' + str(tau))
    ax0.legend(loc='upper right')

    # Attractor
    embedded = embedding(signal, delay=tau, dimension=3)
    x = embedded[:, 0]
    y = embedded[:, 1]
    z = embedded[:, 2]

    ax1 = fig.add_subplot(spec[1])

    #   Chunk the data into colorbars
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(z.min(), z.max())
    lc = matplotlib.collections.LineCollection(segments, cmap='plasma', norm=norm)
    lc.set_array(z)
    line = ax1.add_collection(lc)

    #   Customize
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(x.min(), x.max())
    ax1.set_xlabel("Signal [i]")
    ax1.set_ylabel("Signal [i-" + str(tau) + "]")

    return fig
