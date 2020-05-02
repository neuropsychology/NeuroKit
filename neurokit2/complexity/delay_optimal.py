# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt

from ..stats import cor
from ..signal import signal_findpeaks
from .delay_embedding import delay_embedding
from .mutual_information import mutual_information


def delay_optimal(signal, tau_max=100, method="fraser1986", show=False):
    """Optimal Time Delay (tau)

    The time delay (Tau) is one of the two critical parameters involved in the construction of the time-delay embedding of a signal.

    Several authors suggested different methods to guide the choice of Tau. Fraser and Swinney (1986) suggest using the first local minimum of the mutual information between the delayed and non-delayed time series, effectively identifying a value of tau for which they share the least information. Theiler (1990) suggested to select Tau such that the autocorrelation between the signal and its lagged version at Tau is the closest to 1/e.

    A simple  criterion  to  compute  Tau  was  suggested  by  .  Let  )(τΨ  be  the  autocorrelation  function  (AC)  from  the  time  series  y(t).  Theiler  suggested  to  select  τ  such  that  ./1)(e≅Ψτ  Fraser  and  Swinney (1986)  propose  to  find  the  first  minimum  of  the  Auto Mutual Information  (AMI).  A  novel  method  for  simultaneously  determining  both  m  and τ proposed by  Gautama  et  al.  (2003)  and  it  is  based  on  the  minimum of the Differential Entropy (DE).




    The code is based on http://node99.org/tutorials/ar/, but very unsure of our implementation.
    Please help us by checking-it.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    tau_max : int
        The maximum time delay to test.
    method : str
        Correlation method. Can be one of 'fraser1986'.
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

    # Method
    method = method.lower()
    if method in ["fraser", "fraser1986"]:
        metric = "Mutual Information"
        algorithm = "first local minimum"
    else:
        raise ValueError("NeuroKit error: delay_optimal(): 'method' "
                         "not recognized.")

    # Get metric
    metric_values = _delay_optimal_metric(signal, tau_sequence, metric=metric)

    # Get optimal tau
    optimal = _delay_optimal_select(metric_values, algorithm=algorithm)

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
def _delay_optimal_select(metric_values, algorithm="first local minimum"):

    if algorithm == "first local minimum":
        optimal = signal_findpeaks(-1 * metric_values, relative_height_min=0.1, relative_max=True)["Peaks"][0]

    return optimal



def _delay_optimal_metric(signal, tau_sequence, metric="Mutual Information"):

    values = np.zeros(len(tau_sequence))

    # Loop through taus and compute all scores values
    for i, current_tau in enumerate(tau_sequence):
        embedded = delay_embedding(signal, delay=current_tau, dimension=2)
        if metric == "Mutual Information":
            values[i] = mutual_information(embedded[:,0], embedded[:,1], normalized=True)
        if metric == "Autocorrelation":
            values[i] = cor(embedded[:,0], embedded[:,1])
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
    embedded = delay_embedding(signal, delay=tau, dimension=3)

    ax1 = fig.add_subplot(spec[1])
    ax1.set_xlabel("Signal [i]")
    ax1.set_ylabel("Signal [i-" + str(tau) + "]")
    ax1.plot(embedded[:,0], embedded[:,1], color='#3F51B5')
    return fig
