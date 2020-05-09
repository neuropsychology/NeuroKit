# -*- coding: utf-8 -*-
import numpy as np

from ..misc import range_log

def fractal_dfa(signal, windows=None, overlap=True, integrate=True, order=1):
    """Detrended Fluctuation Analysis (DFA)

    Computes Detrended Fluctuation Analysis (DFA) on the time series data.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    windows : list
        The lengths of the windows (number of data points in each subseries). If None, will set it to a logarithmic scale (so that each window scale hase the same weight) with a minimum of 4 and maximum of a tenth of the length (to have more than 10 windows to calculate the average fluctuation).
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
    >>> signal = nk.signal_simulate(duration=1)
    >>> nk.fractal_dfa(signal)
    1.9619134459122758


    References
    -----------
    - Hardstone, R., Poil, S. S., Schiavone, G., Jansen, R., Nikulin, V. V., Mansvelder, H. D., & Linkenkaer-Hansen, K. (2012). Detrended fluctuation analysis: a scale-free view on neuronal oscillations. Frontiers in physiology, 3, 450.
    - `nolds` <https://github.com/CSchoel/nolds/blob/master/nolds/measures.py>
    """
    # Sanity checks
    n = len(signal)

    if windows is None:
        if n >= 80:
            windows = range_log(4, 0.1 * n, 1.2)  # Default window
        else:
            raise ValueError("NeuroKit error: fractal_dfa(): signal is too short to compute DFA.")

    # Check windows
    if len(windows) < 2:
        raise ValueError("NeuroKit error: fractal_dfa(): more than one window is needed.")
    if np.min(windows) < 2:
        raise ValueError("NeuroKit error: fractal_dfa(): there must be at least 2 data points in each window")
    if np.max(windows) >= n:
        raise ValueError("NeuroKit error: fractal_dfa(): the window cannot contain more data points than the time series.")

    if integrate is True:
        # Determine signal profile
        signal = np.cumsum(signal - np.mean(signal))


    # Divide profile
    fluctuations = []
    for window in windows:
        if overlap:
            d = np.array([signal[i:i + window] for i in range(0, len(signal) - window, window // 2)])
        else:
            d = signal[:N - (N % window)]
            d = d.reshape((signal.shape[0] // window, window))

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
