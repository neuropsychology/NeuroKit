# -*- coding: utf-8 -*-
import numpy as np


def complexity_dfa(signal, windows=None, overlap=True, order=1):
    """
    Computes Detrended Fluctuation Analysis (DFA) on the time series data.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    windows : list
        The length of the data in each subseries. Defaults to None.
    overlap : bool
        Defaults to True, where the windows will have a 50% overlap
        with each other, otherwise non-overlapping windows will be used.
    order : int
        The order of the trend.

    Returns
    ----------
    poly : float
        The estimate alpha of the Hurst parameter.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> nk.complexity_dfa(signal)
    1.9619134459122758


    References
    -----------
    - `nolds` <https://github.com/CSchoel/nolds/blob/master/nolds/measures.py>
    """
    # Sanity-check input
    signal = np.asarray(signal)
    N = len(signal)

    if windows is None:
        if N > 70:
            windows = _complexity_dfa_n(4, 0.1 * N, 1.2)
        elif N > 10:
            windows = [4, 5, 6, 7, 8, 9]
        else:
            windows = [N-2, N-1]
            print("NeuroKit warning: complexity_dfa(): DFA with less than ten data points is unreliable.")

    if len(windows) < 2:
        raise ValueError("NeuroKit error: complexity_dfa(): more than one window is needed.")
    if np.min(windows) < 2:
        raise ValueError("NeuroKit error: complexity_dfa(): there must be at least 2 data points in each window")
    if np.max(windows) >= N:
        raise ValueError("NeuroKit error: complexity_dfa(): the window cannot contain more data points than the time series.")

    # Determine signal profile
    integrated = np.cumsum(signal - np.mean(signal))

    # Divide profile
    fluctuations = []
    for window in windows:
        if overlap:
            d = np.array([integrated[i:i + window] for i in range(0, len(integrated) - window, window // 2)])
        else:
            d = integrated[:N - (N % window)]
            d = d.reshape((integrated.shape[0] // window, window))

        # Local trend
        x = np.arange(window)
        poly = [np.polyfit(x, d[i], order) for i in range(len(d))]
        trend = np.array([np.polyval(poly[i], x) for i in range(len(d))])

        # Calculate fluctuation around trend
        fluctuation = np.sqrt(np.sum((d - trend) ** 2, axis=1) / window)

        # Mean fluctuation
        mean_fluctuation = np.sum(fluctuation) / len(fluctuation)
        fluctuations.append(mean_fluctuation)

    fluctuations = np.array(fluctuations)

    # filter zeros
    nonzero = np.where(fluctuations != 0)
    windows = np.array(windows)[nonzero]

    if len(fluctuations) == 0:
        poly = [np.nan, np.nan]
    else:
        poly = np.polyfit(np.log(windows), np.log(fluctuations), order)

    return poly[0]


# =============================================================================
# Internals
# =============================================================================
def _complexity_dfa_n(start, end, factor):
    """Computes successively a list of integers.
    """
    end_i = int(np.floor(np.log(1.0 * end / start) / np.log(factor)))
    ns = [start]

    for i in range(end_i + 1):
        n = int(np.floor(start * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)

    return ns
