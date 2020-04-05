# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..signal import signal_plot
from .mutual_information import mutual_information


def optimal_delay(signal, tau_max=100, show=False):
    """Optimal Time Delay (tau)

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
    >>> signal = nk.signal_simulate()
    >>> nk.optimal_delay(signal, tau_max=1000, show=True)
    """
    # find usable time delay via mutual information
    values = np.zeros(tau_max)

    optimal = None
    for i, tau in enumerate(range(1, tau_max)):
        unlagged = signal[:-tau]
        lagged = np.roll(signal, -tau)[:-tau]
        values[i] = mutual_information(lagged, unlagged, normalized=True)

        if i > 0 and optimal is None and values[i-1] < values[i]: # return first local minima
            optimal = tau-1

    if show is True:
        signal_plot(values)

    return optimal



