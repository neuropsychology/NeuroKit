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
        window scale has the same weight) with a minimum of 4 and maximum of a tenth of the length
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
        The order of the polynomial detrending, 1 for the linear trend.
    multifractal : bool
        If true, compute Multifractal Detrended Fluctuation Analysis (MFDFA), in which case the argument
        ``q`` is taken into account.
    q : list
        The sequence of fractal exponents when ``multifractal=True``. Must be a sequence between -10
        and 10 (note that zero will be removed, since the code does not converge there). Setting
        q = 2 (default) gives a result of a standard DFA. For instance, Ihlen (2012) uses ``
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

    - `MFDFA <https://github.com/LRydin/MFDFA/>`_

    - `Youtube introduction <https://www.youtube.com/watch?v=o0LndP2OlUI>`_

    """

    # Enforce DFA in case 'multifractal = False' but 'q' is not 2
    if multifractal == False:
        q = 2

    # Sanitize fractal power (cannot be close to 0)
    q = _cleanse_q(q)

    # '_fractal_dfa()' does the heavy lifting,
    windows, fluctuations = _fractal_dfa(signal = signal, windows = windows, overlap = overlap,
                                         integrate = integrate, order = order,
                                         multifractal = multifractal, q = q)

    # Plot if show is True.
    if show is True:
        _fractal_dfa_plot(windows, fluctuations, multifractal, q)

    # if multifractal, then no return, since DFA fit to extract the Hurst coefficient
    # is only la
    if len(fluctuations) == 0:
        return np.nan
    if multifractal == False:
        dfa = np.polyfit(np.log2(windows), np.log2(fluctuations), 1)[0]
    else:
        dfa = np.zeros(len(q))
        for i in range(len(q)):
            dfa[i] = np.polyfit(np.log2(windows), np.log2(fluctuations[:,i]), 1)[0]

    return dfa


# =============================================================================
# Utilities
# =============================================================================

def _fractal_dfa(signal, windows="default", overlap=True, integrate=True, order=1, multifractal=False, q=2):
    """Does the heavy lifting for `fractal_dfa()`.

    Returns
    ----------
    windows : list
        A list containing the lengths of the windows

    fluctuations : np.ndarray
        The detrended fluctuations, from DFA or MFDFA.

    """

    # Sanitize fractal power (cannot be close to 0)
    q = _cleanse_q(q)

    # Sanity checks
    n = len(signal)
    windows = _fractal_dfa_findwindows(n, windows)

    # Preprocessing
    if integrate is True:
        signal = np.cumsum(signal - np.mean(signal))  # Get signal profile

    # Function to store fluctuations. For DFA this is an array. For MFDFA, this is a matrix
    # of size (len(windows),len(q))
    fluctuations = np.zeros((len(windows),len(q)))
    # Start looping over windows
    for i, window in enumerate(windows):

        # Get window
        segments = _fractal_dfa_getwindow(signal, n, window, overlap=overlap)

        # Get polynomial trends
        trends = _fractal_dfa_trends(segments, window, order=order)

        # Get local fluctuation
        fluctuations[i] = _fractal_dfa_fluctuation(segments, trends, multifractal, q)

    # Filter zeros
    nonzero = np.nonzero(fluctuations)[0]
    windows = windows[nonzero]
    fluctuations = fluctuations[nonzero]

    return windows, fluctuations


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
    # This function reshapes the segments from a one-dimensional array to a matrix for faster
    # polynomail fittings. 'Overlap' reshapes into several overlapping partitions of the data
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
        # obtain the fluctuation function, which is a function of the windows and of q
        fluctuation = np.float_power(np.mean(np.float_power(var, q / 2), axis=1), 1 / q.T)

        # Remnant:
        # To recover just the conventional DFA, find q=2
        # fluctuation = np.mean(fluctuation)

    else:
        # Compute Root Mean Square (RMS)
        fluctuation = np.sum(detrended ** 2, axis=1) / detrended.shape[1]
        fluctuation = np.sqrt(np.sum(fluctuation) / len(fluctuation))

    return fluctuation


def _fractal_dfa_plot(windows, fluctuations, multifractal, q):

    if multifractal == False:
        dfa = np.polyfit(np.log2(windows), np.log2(fluctuations), 1)

        fluctfit = 2 ** np.polyval(dfa, np.log2(windows))
        plt.loglog(windows, fluctuations, "bo")
        plt.loglog(windows, fluctfit, "r", label=r"$\alpha$ = {:.3f}".format(dfa[0][0]))
    else:
        for i in range(len(q)):
            dfa = np.polyfit(np.log2(windows), np.log2(fluctuations[:,i]), 1)


            fluctfit = 2 ** np.polyval(dfa, np.log2(windows))

            plt.loglog(windows, fluctuations, "bo")
            plt.loglog(windows, fluctfit, "r", label=r"$\alpha$ = {:.3f}, q={:.1f}".format(dfa[0],q[i][0]))

    plt.title("DFA")
    plt.xlabel(r"$\log_{2}$(Window)")
    plt.ylabel(r"$\log_{2}$(Fluctuation)")
    plt.legend()
    plt.show()


# =============================================================================
#  Utils MFDFA
# =============================================================================

## This is based on Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E.,
# Havlin, S., Bunde, A., & Stanley, H. E., Multifractal detrended fluctuation
# analysis of nonstationary time series. Physica A, 316(1-4), 87-114, 2002 as
# well as on nolds (https://github.com/CSchoel/nolds) and on work by  Espen A.
# F. Ihlen, Introduction to multifractal detrended fluctuation analysis in
# Matlab, Front. Physiol., 2012, https://doi.org/10.3389/fphys.2012.00141
#
# It was designed by Leonardo Rydin Gorjão and is part of MFDFA
# (https://github.com/LRydin/MFDFA). It is included here by the author.


def _cleanse_q(q=2):
    # TODO: Add log calculator for q ≈ 0

    # Fractal powers as floats
    q = np.asarray_chkfinite(q, dtype=np.float)

    # Ensure q≈0 is removed, since it does not converge. Limit set at |q| < 0.1
    q = q[(q < -0.1) + (q > 0.1)]

    # Reshape q to perform np.float_power
    q = q.reshape(-1, 1)

    return q


def singularity_spectrum(signal, q = list(range(-10,10)), lim = [None, None], windows="default", overlap=True,
    integrate=True, order=1, show=False):
    """Extract the slopes of the fluctuation function to futher obtain the the singularity strength
    `hq` (or Hölder exponents) and singularity spectrum `Dq` (or fractal dimension).
    This is iconically shaped as an inverse parabola, but most often it is difficult to obtain the
    negative `q` terms, and one can focus on the left side of the parabola (`q>0`).

    The parameters are mostly identical to `fractal_mfdfa()`, as one needs to perform MFDFA to
    obtain the singularity spectrum. Calculating only the DFA is insufficient, as it only has `q=2`,
    and we need a set of `q` values, thus here we default to
    `q = list(range(-10,10))`, where the `0` element is removed by `_cleanse_q()`.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    q : list (default 'list(range(-10,10))')
        The sequence of fractal exponents. Must be a sequence between -10
        and 10 (note that zero will be removed, since the code does not converge there).
    lim: list (default = [None, None])
        List of lower and upper lag limits. If none, the polynomial fittings
        include the full range of lag.
    windows : list
        A list containing the lengths of the windows (number of data points in each subseries). Also
        referred to as 'lag' or 'scale'. If 'default', will set it to a logarithmic scale (so that each
        window scale has the same weight) with a minimum of 4 and maximum of a tenth of the length
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
        The order of the polynomial detrending, 1 for the linear trend.
    show : bool
        Visualise the singularity plot.

    Returns
    -------
    hq: np.array
        Singularity strength `hq`. The width of this function indicates the
        strength of the multifractality. A width of `max(hq) - min(hq) ≈ 0`
        means the data is monofractal.

    Dq: np.array
        Singularity spectrum `Dq`. The location of the maximum of `Dq` (with
         `hq` as the abscissa) should be 1 and indicates the most prominent
         exponent in the data.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=3, noise=0.05)
    >>> dfa1 = nk.singularity_spectrum(signal, show=True)
    >>> dfa1 #doctest: +SKIP


    References
    -----------
    - Ihlen, E. A. F. E. (2012). Introduction to multifractal detrended fluctuation analysis in Matlab.
      Frontiers in physiology, 3, 141.

    - J. W. Kantelhardt, S. A. Zschiegner, E. Koscielny-Bunde, S. Havlin, A. Bunde, H. E. Stanley
      (2002). Multifractal detrended fluctuation analysis of nonstationary time series. Physica A,
      316(1-4), 87–114.

    - `MFDFA <https://github.com/LRydin/MFDFA/>`_

    """

    q = _cleanse_q(q)

    # obtain the windows and fluctuations
    windows, fluctuations = _fractal_dfa(signal = signal, windows = windows, overlap = overlap,
                                         integrate = integrate, order = order,
                                         multifractal = True, q = q)

    # restructure windows for '_singularity_spectrum()' and associated functions
    windows = windows.reshape(-1, q.size)[:,0]

    hq, Dq = _singularity_spectrum(lag = windows, mfdfa = fluctuations, q = q, lim = lim)

    if show is True:
        singularity_spectrum_plot(hq, Dq)

    return hq, Dq

def _singularity_spectrum(lag: np.array, mfdfa: np.ndarray, q: np.array, show=False,
                          lim: list=[None, None]):

    # Calculate tau
    tau = scaling_exponents(lag, mfdfa, q, lim)

    # Calculate hq, which needs tau
    hq = hurst_exponents(lag, mfdfa, q, lim)

    # Calculate Dq, which needs tau and hq
    Dq = _Dq(tau, hq, q)

    return hq, Dq

def scaling_exponents(lag: np.array, mfdfa: np.ndarray, q: np.array,
                      lim: list=[None, None]):
    """
    Calculate the multifractal scaling exponents ``tau``.

    Parameters
    ----------
    lag: np.array of ints
        An array with the window sizes which where used in MFDFA.

    mdfda: np.ndarray
        Matrix of the fluctuation function from MFDFA

    q: np.array
        Fractal exponents used. Must be more than 2 points.

    lim: list (default = [None, None])
        List of lower and upper lag limits. If none, the polynomial fittings
        include the full range of lag.

    Returns
    -------
    hq: np.array
        Singularity strength ``hq``. The width of this function indicates the
        strength of the multifractality. A width of ``max(hq) - min(hq) ≈ 0``
        means the data is monofractal.

    Dq: np.array
        Singularity spectrum ``Dq``. The location of the maximum of ``Dq`` (with
         ``hq`` as the abscissa) should be 1 and indicates the most prominent
         exponent in the data.

    References
    ----------
    .. [Kantelhardt2002] J. W. Kantelhardt, S. A. Zschiegner, E.
        Koscielny-Bunde, S. Havlin, A. Bunde, H. E. Stanley. "Multifractal
        detrended fluctuation analysis of nonstationary time series." Physica A,
        316(1-4), 87–114, 2002.

    - `MFDFA <https://github.com/LRydin/MFDFA/>`_
    """

    # Calculate the slopes
    slopes = _slopes(lag, mfdfa, q, lim)

    return ( q * slopes ) - 1

def hurst_exponents(lag: np.array, mfdfa: np.ndarray, q: np.array,
                    lim: list=[None, None]):
    """
    Calculate the generalised Hurst exponents 'hq' from the

    Parameters
    ----------
    lag: np.array of ints
        An array with the window sizes which where used in MFDFA.

    mdfda: np.ndarray
        Matrix of the fluctuation function from MFDFA

    q: np.array
        Fractal exponents used. Must be more than 2 points.

    lim: list (default = [None, None])
        List of lower and upper lag limits. If none, the polynomial fittings
        include the full range of lag.

    Returns
    -------
    hq: np.array
        Singularity strength ``hq``. The width of this function indicates the
        strength of the multifractality. A width of ``max(hq) - min(hq) ≈ 0``
        means the data is monofractal.

    References
    ----------
    .. [Kantelhardt2002] J. W. Kantelhardt, S. A. Zschiegner, E.
        Koscielny-Bunde, S. Havlin, A. Bunde, H. E. Stanley. "Multifractal
        detrended fluctuation analysis of nonstationary time series." Physica A,
        316(1-4), 87–114, 2002.

    - `MFDFA <https://github.com/LRydin/MFDFA/>`_
    """

    # Calculate tau
    tau = scaling_exponents(lag, mfdfa, q, lim)

    return ( np.gradient(tau) / np.gradient(q) )

def _slopes(lag: np.array, mfdfa: np.ndarray, q: np.array,
            lim: list=[None, None]):
    """
    Extra the slopes of each q power obtained with MFDFA to later produce either
    the singularity spectrum or the multifractal exponents.

    References
    ----------
    - `MFDFA <https://github.com/LRydin/MFDFA/>`_

    """

    # Fractal powers as floats
    q = np.asarray_chkfinite(q, dtype = float)

    # Ensure mfdfa has the same q-power entries as q
    if mfdfa.shape[1] != q.shape[0]:
        raise ValueError(
            "Fluctuation function and q powers don't match in dimension.")

    # Allocated array for slopes
    slopes = np.zeros(len(q))

    # Find slopes of each q-power
    for i in range(len(q)):
        slopes[i] = np.polyfit(
                        np.log(lag[lim[0]:lim[1]]),
                        np.log(mfdfa[lim[0]:lim[1],i]),
                        1)[1]

    return slopes

def _Dq(tau, hq, q):
    """
    Calculate the singularity spectrum or fractal dimension ``Dq``.

    References
    ----------
    - `MFDFA <https://github.com/LRydin/MFDFA/>`_
    """

    return q * hq - tau

def singularity_spectrum_plot(hq, Dq):
    """
    Plots the singularity spectrum.

    Parameters
    ----------
    hq: np.array
        Singularity strength `hq` as calculated with `singularity_spectrum`.

    Dq: np.array
        Singularity spectrum `Dq` as calculated with `singularity_spectrum`.

    Returns
    -------
    fig: matplotlib fig
        Returns the figure, useful if one wishes to use fig.savefig(...).

    References
    ----------
    - `MFDFA <https://github.com/LRydin/MFDFA/>`_
    """

    plt.plot(hq, Dq)

    plt.title("Singularity Spectrum")
    plt.set_ylabel(r'Dq')
    plt.set_xlabel(r'hq')
    # plt.legend()
    plt.show()

    return None

def scaling_exponents_plot(q, tau):
    """
    Plots the scaling exponents, which is conventionally given with `q` in the
    abscissa and `tau` in the ordinates.

    Parameters
    ----------
    q: np.array
        q powers.

    tau: np.array
        Scaling exponents `tau` as calculated with `scaling_exponents`.

    References
    -----------
    - `MFDFA <https://github.com/LRydin/MFDFA/>`_

    """

    plt.plot(q, tau)

    plt.title("Scaling Exponents")
    plt.set_ylabel(r'tau')
    plt.set_xlabel(r'q')
    # plt.legend()
    plt.show()

    return None

def hurst_exponents_plot(q, hq):
    """
    Plots the generalised Hurst exponents with `q` in the abscissa and
    `hq` in the ordinates.

    Parameters
    ----------
    q: np.array
        q powers.

    hq: np.array
        Generalised Hurst coefficients `hq` as calculated with
        `hurst_exponents()`.

    References
    -----------
    - `MFDFA <https://github.com/LRydin/MFDFA/>`_

    """

    plt.plot(q, tau)

    plt.title("Generalised Hurst Exponents")
    plt.set_ylabel(r'hq')
    plt.set_xlabel(r'q')
    # plt.legend()
    plt.show()

    return None
