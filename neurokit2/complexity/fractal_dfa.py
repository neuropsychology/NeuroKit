# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from ..misc import expspace


def fractal_dfa(signal, windows="default", overlap=True, integrate=True, order=1, multifractal=False, q=2, show=False):
    """(Multifractal) Detrended Fluctuation Analysis (DFA or MFDFA)

    Python implementation of Detrended Fluctuation Analysis (DFA) or Multifractal DFA of a signal.
    Detrended fluctuation analysis, much like the Hurst exponent, is used to find long-term statistical
    dependencies in time series.

    This function can be called either via ``fractal_dfa()`` or ``complexity_dfa()``, and its multifractal
    variant can be directly accessed via ``fractal_mfdfa()`` or ``complexity_mfdfa()``

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    windows : list
        A list containing the lengths of the windows (number of data points in each subseries). Also
        referred to as 'lag' or 'scale'. If 'default', will set it to a logarithmic scale (so that each
        window scale hase the same weight) with a minimum of 4 and maximum of a tenth of the length
        (to have more than 10 windows to calculate the average fluctuation).
    overlap : bool
        Defaults to True, where the windows will have a 50% overlap
        with each other, otherwise non-overlapping windows will be used.
    integrate : bool
        It is common practice to convert the signal to a random walk (i.e., detrend and integrate,
        which corresponds to the signal 'profile'). Note that it leads to the flattening of the signal,
        which can lead to the loss of some details (see Ihlen, 2012 for an explanation). Note that for
        strongly anticorrelated signals, this transformation should be applied two times (i.e., provide
        ``np.cumsum(signal - np.mean(signal))`` instead of ``signal``).
    order : int
        The order of the polynoiam trend, 1 for the linear trend.
    multifractal : bool
        If true, compute Multifractal Detrended Fluctuation Analysis (MFDFA), in which case the argument
        ```q`` is taken into account.
    q : list
        The sequence of fractal exponents when ``multifractal=True``. Must be a sequence between -10
        and 10 (nota that zero will be removed, since the code does not converge there). Setting
        q = 2 (default) gives a result close to a standard DFA. For instance, Ihlen (2012) usese ``
        q=[-5, -3, -1, 0, 1, 3, 5]``.
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
    >>> signal = nk.signal_simulate(duration=3, noise=0.05)
    >>> dfa1 = nk.fractal_dfa(signal, show=True)
    >>> dfa1 #doctest: +SKIP
    >>> dfa2 = nk.fractal_mfdfa(signal, q=np.arange(-3, 4), show=True)
    >>> dfa2 #doctest: +SKIP


    References
    -----------
    - Ihlen, E. A. F. E. (2012). Introduction to multifractal detrended fluctuation analysis in Matlab.
      Frontiers in physiology, 3, 141.

    - Hardstone, R., Poil, S. S., Schiavone, G., Jansen, R., Nikulin, V. V., Mansvelder, H. D., &
      Linkenkaer-Hansen, K. (2012). Detrended fluctuation analysis: a scale-free view on neuronal
      oscillations. Frontiers in physiology, 3, 450.

    - `nolds <https://github.com/CSchoel/nolds/>`_

    - `Youtube introduction <https://www.youtube.com/watch?v=o0LndP2OlUI>`_

    """
    # Sanity checks
    n = len(signal)
    windows = _fractal_dfa_findwindows(n, windows)

    # Preprocessing
    if integrate is True:
        signal = np.cumsum(signal - np.mean(signal))  # Get signal profile

    # Sanitize fractal power
    if multifractal is True:
        q = _fractal_mfdfa_q(q)

    fluctuations = np.zeros(len(windows))
    # Start looping over windows
    for i, window in enumerate(windows):

        # Get window
        segments = _fractal_dfa_getwindow(signal, n, window, overlap=overlap)

        # Get polynomial trends
        trends = _fractal_dfa_trends(segments, window, order=1)

        # Get local fluctuation
        fluctuations[i] = _fractal_dfa_fluctuation(segments, trends, multifractal, q)

    # Filter zeros
    nonzero = np.nonzero(fluctuations)[0]
    windows = windows[nonzero]
    fluctuations = fluctuations[nonzero]

    # Compute trend
    if len(fluctuations) == 0:
        return np.nan
    else:
        dfa = np.polyfit(np.log2(windows), np.log2(fluctuations), order)

    if show is True:
        _fractal_dfa_plot(windows, fluctuations, dfa)

    return dfa[0]


# =============================================================================
# Utilities
# =============================================================================


def _fractal_dfa_findwindows(n, windows="default"):
    # Convert to array
    if isinstance(windows, list):
        windows = np.asarray(windows)

    # Default windows number
    if windows is None or isinstance(windows, str):
        windows = np.int(n / 10)

    # Default windows sequence
    if isinstance(windows, int):
        windows = expspace(
            10, np.int(n / 10), windows, base=2
        )  # see https://github.com/neuropsychology/NeuroKit/issues/206
        windows = np.unique(windows)  # keep only unique

    # Check windows
    if len(windows) < 2:
        raise ValueError("NeuroKit error: fractal_dfa(): more than one window is needed.")
    if np.min(windows) < 2:
        raise ValueError("NeuroKit error: fractal_dfa(): there must be at least 2 data points" "in each window")
    if np.max(windows) >= n:
        raise ValueError(
            "NeuroKit error: fractal_dfa(): the window cannot contain more data points than the" "time series."
        )
    return windows


def _fractal_dfa_getwindow(signal, n, window, overlap=True):
    if overlap:
        segments = np.array([signal[i : i + window] for i in np.arange(0, n - window, window // 2)])
    else:
        segments = signal[: n - (n % window)]
        segments = segments.reshape((signal.shape[0] // window, window))
    return segments


def _fractal_dfa_trends(segments, window, order=1):
    x = np.arange(window)

    coefs = np.polyfit(x[:window], segments.T, order).T

    # TODO: Could this be optimized? Something like np.polyval(x[:window], coefs)
    trends = np.array([np.polyval(coefs[j], x) for j in np.arange(len(segments))])

    return trends


def _fractal_dfa_fluctuation(segments, trends, multifractal=False, q=2):

    detrended = segments - trends

    if multifractal is True:
        var = np.var(detrended, axis=1)
        fluctuation = np.float_power(np.mean(np.float_power(var, q / 2), axis=1) / 2, 1 / q.T)
        fluctuation = np.mean(fluctuation)  # Average over qs (not sure of that!)

    else:
        # Compute Root Mean Square (RMS)
        fluctuation = np.sum(detrended ** 2, axis=1) / detrended.shape[1]
        fluctuation = np.sqrt(np.sum(fluctuation) / len(fluctuation))

    return fluctuation


def _fractal_dfa_plot(windows, fluctuations, dfa):
    fluctfit = 2 ** np.polyval(dfa, np.log2(windows))
    plt.loglog(windows, fluctuations, "bo")
    plt.loglog(windows, fluctfit, "r", label=r"$\alpha$ = %0.3f" % dfa[0])
    plt.title("DFA")
    plt.xlabel(r"$\log_{2}$(Window)")
    plt.ylabel(r"$\log_{2}$(Fluctuation)")
    plt.legend()
    plt.show()


# =============================================================================
#  Utils MDDFA
# =============================================================================


def _fractal_mfdfa_q(q=2):
    # TODO: Add log calculator for q ≈ 0

    # Fractal powers as floats
    q = np.asarray_chkfinite(q, dtype=np.float)

    # Ensure q≈0 is removed, since it does not converge. Limit set at |q| < 0.1
    q = q[(q < -0.1) + (q > 0.1)]

    # Reshape q to perform np.float_power
    q = q.reshape(-1, 1)
    return q
