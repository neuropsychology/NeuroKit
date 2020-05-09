# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from ..misc import range_log


def fractal_dfa(signal, windows="default", overlap=True, integrate=True, order=1, show=False, **kwargs):
    """Detrended Fluctuation Analysis (DFA)

    Computes Detrended Fluctuation Analysis (DFA) on the time series data. Detrended fluctuation analysis, much like the Hurst exponent, is used to
    find long-term statistical dependencies in time series.

    This function can be called either via ``fractal_dfa()`` or ``complexity_dfa()``.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    windows : list
        The lengths of the windows (number of data points in each subseries). If 'default', will set it to a logarithmic scale (so that each window scale hase the same weight) with a minimum of 4 and maximum of a tenth of the length (to have more than 10 windows to calculate the average fluctuation).
    overlap : bool
        Defaults to True, where the windows will have a 50% overlap
        with each other, otherwise non-overlapping windows will be used.
    integrate : bool
        It is common practice to integrate the signal (so that the resulting set can be interpreted in the framework of a random walk). Note that it leads to the flattening of the signal, which can lead to the loss of some details.
    order : int
        The order of the trend, 1 for linear.
    show : bool
        Visualise the trend between the window size and the fluctuations.

    Returns
    ----------
    dfa : float
        The DFA coefficient.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=1)
    >>> nk.fractal_dfa(signal)[2], windows=[10, 20, 40, 70])


    References
    -----------
    - Hardstone, R., Poil, S. S., Schiavone, G., Jansen, R., Nikulin, V. V., Mansvelder, H. D., & Linkenkaer-Hansen, K. (2012). Detrended fluctuation analysis: a scale-free view on neuronal oscillations. Frontiers in physiology, 3, 450.
    - `nolds <https://github.com/CSchoel/nolds/>`_
    - `Youtube introduction <https://www.youtube.com/watch?v=o0LndP2OlUI>`_
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

        # Mean fluctuation
        fluctuations[i] = _fractal_dfa_fluctuation(segments, window, order=order, **kwargs)


    # Filter zeros
    nonzero = np.nonzero(fluctuations)[0]
    windows = windows[nonzero]
    fluctuations = fluctuations[nonzero]

    # Compute trend
    if len(fluctuations) == 0:
        return np.nan
    else:
        dfa = np.polyfit(np.log(windows), np.log(fluctuations), order)

    if show is True:
        _fractal_dfa_plot(windows, fluctuations, dfa)

    return dfa[0]


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



def _fractal_dfa_findwindows(signal, n, windows='default'):
    # Convert to array
    if isinstance(windows, list):
        windows = np.asarray(windows)

    # Default windows
    if windows is None or isinstance(windows, str):
        if n >= 80:
            windows = range_log(4, 0.1 * n, 1.2)  # Default window
            # scale_lim=[5,9], scale_dens=0.25; scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
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



def _fractal_dfa_plot(windows, fluctuations, dfa):
    fluctfit = 2**np.polyval(dfa, np.log2(windows))
    plt.loglog(windows, fluctuations, 'bo')
    plt.loglog(windows, fluctfit, 'r', label=r'$\alpha$ = %0.2f' %dfa[0])
    plt.title('DFA')
    plt.xlabel(r'$\log_{10}$(time window)')
    plt.ylabel(r'$\log_{10}$<F(t)>')
    plt.legend()
    plt.show()



def _fractal_dfa_fluctuation(segments, window, order=1, rms=False):
    # Local trend
    x = np.arange(window)
    j_segments = np.arange(len(segments))

    poly = np.array([np.polyfit(x, segments[j], order) for j in j_segments])
    trend = np.array([np.polyval(poly[j], x) for j in j_segments])

    if rms is True:
        # Method from https://github.com/dokato/dfa/blob/master/dfa.py by Dominik Krzeminski
        rms = np.sqrt(np.mean((segments - trend)**2))
        fluctuation = np.sqrt(np.mean(rms**2))
    else:
        # Calculate fluctuation around trend
        fluctuation = np.sqrt(np.sum((segments - trend) ** 2, axis=1) / window)
        # Mean fluctuation
        fluctuation = np.sum(fluctuation) / len(fluctuation)
    return fluctuation
