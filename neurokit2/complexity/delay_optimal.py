# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt

from ..signal import signal_findpeaks
from .delay_embedding import delay_embedding
from .mutual_information import mutual_information


def delay_optimal(signal, tau_max=100, method="firstlocalminimum", show=False):
    """Optimal Time Delay (tau)

    Most of the complexity indices for time series require the user to specify "Tau", .
    Fraser and Swinney (1986) suggest using the first local minimum of the mutual information between the delayed and non-delayed time series, effectively identifying a value of tau for which they share the least information.

    The code is based on http://node99.org/tutorials/ar/, but very unsure of our implementation.
    Please help us by checking-it.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    tau_max : int
        The maximum time delay to test.
    show : bool
        If true, will plot the mutual information values for each value of tau.

    Returns
    -------
    int
        Optimal time delay.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=10, frequency=1, noise=0.01)
    >>> nk.signal_plot(signal)
    >>>
    >>> tau = nk.delay_optimal(signal, tau_max=1000, show=True)
    >>>
    >>> # Realistic example
    >>> ecg = nk.ecg_simulate(duration=120, sampling_rate=200)
    >>> signal = nk.ecg_rate(nk.ecg_peaks(ecg, sampling_rate=200)[0], sampling_rate=200)
    >>> nk.signal_plot(signal)
    >>>
    >>> tau = nk.delay_optimal(signal, tau_max=1000, show=True)
    """
    # Initalize vectors
    if isinstance(tau_max, int):
        tau_sequence = np.arange(1, tau_max)
    else:
        tau_sequence = np.array(tau_max)
    values = np.zeros(len(tau_sequence))

    # Loop through taus and compute all MI values
    for i, current_tau in enumerate(tau_sequence):
        embedded = delay_embedding(signal, delay=current_tau, dimension=2)
        values[i] = mutual_information(embedded[:,0], embedded[:,1], normalized=True)

    # Get optimal tau
    if method in ["localmin", "min", "firstlocalminimum", "firstlocalmin"]:
        optimal = signal_findpeaks(-1*values, relative_height_min=0.1, relative_max=True)["Peaks"][0]
    else:
        raise ValueError("NeuroKit error: delay_optimal(): 'method' "
                         "not recognized.")

    if show is True:
        _optimal_delay_plot(signal, mutual_information=values, tau_sequence=tau_sequence, tau=optimal)


    return optimal




def _optimal_delay_plot(signal, mutual_information, tau_sequence, tau=1):
    """
    """
    fig = plt.figure(constrained_layout=False)
    spec = matplotlib.gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 3], width_ratios=[2])

    ax0 = fig.add_subplot(spec[0])
    ax0.set_xlabel("Time Delay (tau)")
    ax0.set_ylabel("Mutual Information")
    ax0.plot(tau_sequence, mutual_information)

    ax1 = fig.add_subplot(spec[1])
    ax1.set_xlabel("Signal[i]")
    ax1.set_ylabel("Signal[i-" + str(tau) + "]")
    ax1.plot(signal[:-1].flatten(), np.roll(signal, -tau)[:-1].flatten())
    return fig
