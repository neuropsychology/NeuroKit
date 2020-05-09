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
    integrate : bool
        It is common practice to integrate the signal (so that the resulting set can be interpreted in the framework of a random walk), altough it leads to the flattening of the signal, that can lead to the loss of some details.
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
    2.0262713695244083


    References
    -----------
    - Hardstone, R., Poil, S. S., Schiavone, G., Jansen, R., Nikulin, V. V., Mansvelder, H. D., & Linkenkaer-Hansen, K. (2012). Detrended fluctuation analysis: a scale-free view on neuronal oscillations. Frontiers in physiology, 3, 450.
    - `nolds` <https://github.com/CSchoel/nolds/blob/master/nolds/measures.py>
    """
    # Sanity checks
    n = len(signal)
    windows = _fractal_dfa_findwindows(signal, n, windows)

    # Preprocessing
    if integrate is True:
        signal = np.cumsum(signal - np.mean(signal))  # Determine signal profile

    # Divide profile
    fluctuations = np.zeros(len(windows))
    for i, window in enumerate(windows):
        # Get window
        segments = _fractal_dfa_getwindow(signal, n, window, overlap=overlap)

        # Local trend
        x = np.arange(window)
        j_segments = np.arange(len(segments))

        poly = np.array([np.polyfit(x, segments[j], order) for j in j_segments])
        trend = np.array([np.polyval(poly[j], x) for j in j_segments])

        # Calculate fluctuation around trend
        fluctuation = np.sqrt(np.sum((segments - trend) ** 2, axis=1) / window)

        # Mean fluctuation
        mean_fluctuation = np.sum(fluctuation) / len(fluctuation)
        fluctuations[i] = mean_fluctuation

    # Filter zeros
    nonzero = np.nonzero(fluctuations)[0]
    windows = windows[nonzero]
    fluctuations = fluctuations[nonzero]

    # Compute trend
    if len(fluctuations) == 0:
        dfa = np.nan
    else:
        dfa = np.polyfit(np.log(windows), np.log(fluctuations), order)[0]

    return dfa


# =============================================================================
# Internals
# =============================================================================
def _fractal_dfa_getwindow(signal, n, window, overlap=True):
    if overlap:
        segments = np.array([signal[i:i + window] for i in np.arange(0, n - window, window // 2)])
    else:
        segments = signal[:n - (n % window)]
        segments = segments.reshape((signal.shape[0] // window, window))
    return segments



def _fractal_dfa_findwindows(signal, n, windows):
    # Default windows
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
    return windows
