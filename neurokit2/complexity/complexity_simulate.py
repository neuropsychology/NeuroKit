# -*- coding: utf-8 -*-
import numpy as np


def complexity_simulate(duration=10, sampling_rate=1000):
    """Simulate chaotic time series.

    Generates time series using the discrete approximation of the
    Mackey-Glass delay differential equation described by Grassberger &
    Procaccia (1983).

    Parameters
    ----------
    duration : int
        Desired length of duration (s).
    sampling_rate, length : int
        The desired sampling rate (in Hz, i.e., samples/second) or the desired

    Examples
    ------------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.complexity_simulate(duration=30, sampling_rate=100)
    >>> nk.signal_plot(signal)

    Returns
    -------
    x : array
        Array containing the time series.
    """
    signal = _complexity_simulate_mackeyglass(duration=duration, sampling_rate=sampling_rate)
    return signal

# =============================================================================
# Methods
# =============================================================================
def _complexity_simulate_mackeyglass(duration=10, sampling_rate=1000, x0=None, a=0.2, b=0.1, c=10.0, n=1000, discard=250):
    """Generate time series using the Mackey-Glass equation.
    Generates time series using the discrete approximation of the
    Mackey-Glass delay differential equation described by Grassberger &
    Procaccia (1983).

    Taken from nolitsa (https://github.com/manu-mannattil/nolitsa/blob/master/nolitsa/data.py#L223).

    Parameters
    ----------
    duration : int
        Duration of the time series to be generated.
    sampling_rate : float, optional (default = 0.46)
        Sampling step of the time series.  It is useful to pick
        something between tau/100 and tau/10, with tau/sampling_rate being
        a factor of n.  This will make sure that there are only whole
        number indices.
    x0 : array, optional (default = random)
        Initial condition for the discrete map.  Should be of length n.
    a : float, optional (default = 0.2)
        Constant a in the Mackey-Glass equation.
    b : float, optional (default = 0.1)
        Constant b in the Mackey-Glass equation.
    c : float, optional (default = 10.0)
        Constant c in the Mackey-Glass equation.
    tau : float, optional (default = 23.0)
        Time delay in the Mackey-Glass equation.
    n : int, optional (default = 1000)
        The number of discrete steps into which the interval between
        t and t + tau should be divided. This results in a time
        step of tau/n and an n + 1 dimensional map.
    discard : int, optional (default = 250)
        Number of n-steps to discard in order to eliminate transients.
        A total of n*discard steps will be discarded.

    """
    length = duration * sampling_rate
    tau = sampling_rate / 2 * 100
    sampling_rate = int(n * sampling_rate / tau)
    grids = n * discard + sampling_rate * length
    x = np.empty(grids)

    if not x0:
        x[:n] = 0.5 + 0.05 * (-1 + 2 * np.random.random(n))
    else:
        x[:n] = x0

    A = (2 * n - b * tau) / (2 * n + b * tau)
    B = a * tau / (2 * n + b * tau)

    for i in range(n - 1, grids - 1):
        x[i + 1] = A * x[i] + B * (x[i - n] / (1 + x[i - n] ** c) +
                                   x[i - n + 1] / (1 + x[i - n + 1] ** c))
    return x[n * discard::sampling_rate]