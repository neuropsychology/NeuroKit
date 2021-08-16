# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from ..misc import expspace


def fractal_dfa(signal, windows="default", overlap=True, integrate=True,
                order=1, multifractal=False, q=2, show=False):
    """(Multifractal) Detrended Fluctuation Analysis (DFA or MFDFA).

    Python implementation of Detrended Fluctuation Analysis (DFA) or
    Multifractal DFA of a signal. Detrended fluctuation analysis, much like the
    Hurst exponent, is used to find long-term statistical dependencies in time
    series.

    This function can be called either via `fractal_dfa()` or
    `complexity_dfa()`, and its multifractal variant can be directly accessed
    via `fractal_mfdfa()` or `complexity_mfdfa()`.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    windows : list
        A list containing the lengths of the windows (number of data points in
        each subseries). Also referred to as 'lag' or 'scale'. If 'default',
        will set it to a logarithmic scale (so that each window scale has the
        same weight) with a minimum of 4 and maximum of a tenth of the length
        (to have more than 10 windows to calculate the average fluctuation).

    overlap : bool
        Defaults to True, where the windows will have a 50% overlap
        with each other, otherwise non-overlapping windows will be used.

    integrate : bool
        It is common practice to convert the signal to a random walk (i.e.,
        detrend and integrate, which corresponds to the signal 'profile'). Note
        that it leads to the flattening of the signal, which can lead to the
        loss of some details (see Ihlen, 2012 for an explanation). Note that for
        strongly anticorrelated signals, this transformation should be applied
        two times (i.e., provide `np.cumsum(signal - np.mean(signal))` instead
        of `signal`).

    order : int
        The order of the polynomial detrending, 1 for the linear trend.

    multifractal : bool
        If true, compute Multifractal Detrended Fluctuation Analysis (MFDFA), in
        which case the argument `q` is taken into account.

    q : list or np.array (default `2`)
        The sequence of fractal exponents when `multifractal=True`. Must be a
        sequence between `-10` and `10` (note that zero will be removed, since
        the code does not converge there). Setting `q = 2` (default) gives a
        result of a standard DFA. For instance, Ihlen (2012) uses
        `q = [-5, -3, -1, 0, 1, 3, 5]`.

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
    - Ihlen, E. A. F. E. (2012). Introduction to multifractal detrended
      fluctuation analysis in Matlab. Frontiers in physiology, 3, 141.

    - Hardstone, R., Poil, S. S., Schiavone, G., Jansen, R., Nikulin, V. V.,
      Mansvelder, H. D., & Linkenkaer-Hansen, K. (2012). Detrended fluctuation
      analysis: a scale-free view on neuronal oscillations. Frontiers in
      physiology, 3, 450.

    - `nolds <https://github.com/CSchoel/nolds/>`_

    - `MFDFA <https://github.com/LRydin/MFDFA/>`_

    - `Youtube introduction <https://www.youtube.com/watch?v=o0LndP2OlUI>`_

    """

    # Enforce DFA in case 'multifractal = False' but 'q' is not 2
    if multifractal is False:
        q = 2

    # Sanitize fractal power (cannot be close to 0)
    q = _cleanse_q(q)

    # obtain the windows and fluctuations
    windows, fluctuations = _fractal_dfa(signal=signal,
                                         windows=windows,
                                         overlap=overlap,
                                         integrate=integrate,
                                         order=order,
                                         multifractal=multifractal,
                                         q=q
                                         )

    # Plot if show is True.
    if show is True:
        _fractal_dfa_plot(windows, fluctuations, multifractal, q)

    if len(fluctuations) == 0:
        return np.nan
    if multifractal is False:
        dfa = np.polyfit(np.log2(windows), np.log2(fluctuations), 1)[0]
    else:
        # Allocated array for slopes
        dfa = np.zeros(len(q))

        # Find slopes of each q-power
        for i in range(len(q)):
            dfa[i] = np.polyfit(np.log2(windows),
                                np.log2(fluctuations[:, i]),
                                1
                                )[0]

    return dfa

# =============================================================================
# Utilities
# =============================================================================


def _fractal_dfa(signal, windows="default", overlap=True, integrate=True,
                 order=1, multifractal=False, q=2):
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

    # Function to store fluctuations. For DFA this is an array. For MFDFA, this
    # is a matrix of size (len(windows),len(q))
    fluctuations = np.zeros((len(windows), len(q)))

    # Start looping over windows
    for i, window in enumerate(windows):

        # Get window
        segments = _fractal_dfa_getwindow(signal, n, window, overlap=overlap)

        # Get polynomial trends
        trends = _fractal_dfa_trends(segments, window, order=order)

        # Get local fluctuation
        fluctuations[i] = _fractal_dfa_fluctuation(segments,
                                                   trends,
                                                   multifractal,
                                                   q
                                                   )

    # I would not advise this part. I understand the need to remove zeros, but I
    # would instead suggest masking it with numpy.ma masked arrays. Note that
    # when 'q' is a list,  windows[nonzero] increases in size.

    # Filter zeros
    # nonzero = np.nonzero(fluctuations)[0]
    # windows = windows[nonzero]
    # fluctuations = fluctuations[nonzero]

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
        raise ValueError(
            "NeuroKit error: fractal_dfa(): more than one window is needed."
        )

    if np.min(windows) < 2:
        raise ValueError(
            "NeuroKit error: fractal_dfa(): there must be at least 2 data "
            "points in each window"
        )
    if np.max(windows) >= n:
        raise ValueError(
            "NeuroKit error: fractal_dfa(): the window cannot contain more data"
            " points than the" "time series."
        )
    return windows


def _fractal_dfa_getwindow(signal, n, window, overlap=True):
    # This function reshapes the segments from a one-dimensional array to a
    # matrix for faster polynomail fittings. 'Overlap' reshapes into several
    # overlapping partitions of the data

    if overlap:
        segments = np.array([signal[i : i + window]
                             for i in np.arange(0, n - window, window // 2)
                             ])
    else:
        segments = signal[: n - (n % window)]
        segments = segments.reshape((signal.shape[0] // window, window))

    return segments


def _fractal_dfa_trends(segments, window, order=1):
    x = np.arange(window)

    coefs = np.polyfit(x[:window], segments.T, order).T

    # TODO: Could this be optimized? Something like np.polyval(x[:window],coefs)
    trends = np.array([np.polyval(coefs[j], x)
                       for j in np.arange(len(segments))
                       ])

    return trends


def _fractal_dfa_fluctuation(segments, trends, multifractal=False, q=2):

    detrended = segments - trends

    if multifractal is True:
        var = np.var(detrended, axis=1)
        # obtain the fluctuation function, which is a function of the windows
        # and of q
        fluctuation = \
            np.float_power(np.mean(np.float_power(var, q / 2), axis=1), 1 / q.T)

        # Remnant:
        # To recover just the conventional DFA, find q=2
        # fluctuation = np.mean(fluctuation)

    else:
        # Compute Root Mean Square (RMS)
        fluctuation = np.sum(detrended ** 2, axis=1) / detrended.shape[1]
        fluctuation = np.sqrt(np.sum(fluctuation) / len(fluctuation))

    return fluctuation


def _fractal_dfa_plot(windows, fluctuations, multifractal, q):

    if multifractal is False:
        dfa = np.polyfit(np.log2(windows), np.log2(fluctuations), 1)

        fluctfit = 2 ** np.polyval(dfa, np.log2(windows))
        plt.loglog(windows, fluctuations, "bo")
        plt.loglog(windows, fluctfit, "r",
                   label=r"$\alpha$ = {:.3f}".format(dfa[0][0]))
    else:
        for i in range(len(q)):
            dfa = np.polyfit(np.log2(windows), np.log2(fluctuations[:, i]), 1)

            fluctfit = 2 ** np.polyval(dfa, np.log2(windows))

            plt.loglog(windows, fluctuations, "bo")
            plt.loglog(windows, fluctfit, "r",
                       label=(r"$\alpha$ = {:.3f}, q={:.1f}"
                              ).format(dfa[0], q[i][0])
                       )

    plt.title(r"DFA")
    plt.xlabel(r"$\log_{2}$(Window)")
    plt.ylabel(r"$\log_{2}$(Fluctuation)")
    plt.legend()
    plt.show()


# =============================================================================
#  Utils MFDFA
# =============================================================================

# This is based on Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E.,
# Havlin, S., Bunde, A., & Stanley, H. E., Multifractal detrended fluctuation
# analysis of nonstationary time series. Physica A, 316(1-4), 87-114, 2002 as
# well as on nolds (https://github.com/CSchoel/nolds) and on work by  Espen A.
# F. Ihlen, Introduction to multifractal detrended fluctuation analysis in
# Matlab, Front. Physiol., 2012, https://doi.org/10.3389/fphys.2012.00141
#
# It was designed by Leonardo Rydin Gorjão as part of MFDFA
# (https://github.com/LRydin/MFDFA). It is included here by the author and
# altered to fit NK to the best of its extent.


def singularity_spectrum(signal, q="default", lim=[None, None],
                         windows="default", overlap=True, integrate=True,
                         order=1, show=False):
    """Extract the slopes of the fluctuation function to futher obtain the
    singularity strength `α` (or Hölder exponents) and singularity spectrum
    `f(α)` (or fractal dimension). This is iconically shaped as an inverse
    parabola, but most often it is difficult to obtain the negative `q` terms,
    and one can focus on the left side of the parabola (`q>0`).

    Note that these measures rarely match the theoretical expectation,
    thus a variation of ± 0.25 is absolutely reasonable.

    The parameters are mostly identical to `fractal_mfdfa()`, as one needs to
    perform MFDFA to obtain the singularity spectrum. Calculating only the
    DFA is insufficient, as it only has `q=2`, and a set of `q` values are
    needed. Here defaulted to `q = list(range(-5,5))`, where the `0` element
    is removed by `_cleanse_q()`.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    q : list or np.array (default `np.linspace(-10,10,41)`)
        The sequence of fractal exponents. Must be a sequence between -10
        and 10 (note that zero will be removed, since the code does not converge
        there). If "default", will takes the form `np.linspace(-10,10,41)`.

    lim: list (default `[1, lag.size//2]`)
        List of lower and upper lag limits. If none, the polynomial fittings
        will be restrict to half the maximal lag and discard the first lag
        point.

    windows : list
        A list containing the lengths of the windows (number of data points in
        each subseries). Also referred to as 'lag' or 'scale'. If 'default',
        will set it to a logarithmic scale (so that each window scale has the
        same weight) with a minimum of 4 and maximum of a tenth of the length
        (to have more than 10 windows to calculate the average fluctuation).

    overlap : bool
        Defaults to True, where the windows will have a 50% overlap
        with each other, otherwise non-overlapping windows will be used.

    integrate : bool
        It is common practice to convert the signal to a random walk (i.e.,
        detrend and integrate, which corresponds to the signal 'profile'). Note
        that it leads to the flattening of the signal, which can lead to the
        loss of some details (see Ihlen, 2012 for an explanation). Note that for
        strongly anticorrelated signals, this transformation should be applied
        two times (i.e., provide `np.cumsum(signal - np.mean(signal))` instead
        of `signal`).

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
    >>> alpha, f = nk.singularity_spectrum(signal, show=True)
    >>> alpha, f #doctest: +SKIP

    References
    -----------
    - Ihlen, E. A. F. E. (2012). Introduction to multifractal detrended
      fluctuation analysis in Matlab. Frontiers in physiology, 3, 141.

    - J. W. Kantelhardt, S. A. Zschiegner, E. Koscielny-Bunde, S. Havlin, A.
      Bunde, H. E. Stanley (2002). Multifractal detrended fluctuation analysis
      of nonstationary time series. Physica A, 316(1-4), 87–114.

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """

    # Draw q values
    if isinstance(q, str):
        q = list(np.linspace(-10, 10, 41))

    q = _cleanse_q(q)

    # obtain the windows and fluctuations
    windows, fluctuations = _fractal_dfa(signal=signal,
                                         windows=windows,
                                         overlap=overlap,
                                         integrate=integrate,
                                         order=order,
                                         multifractal=True,
                                         q=q
                                         )

    # flatten q for the multiplications ahead
    q = q.flatten()

    # if no limits given
    if lim[0] is None and lim[1] is None:
        lim = [1, windows.size // 2]

    # Calculate the slopes
    slopes = _slopes(windows, fluctuations, q, lim)

    # Calculate τ
    tau = q * slopes - 1

    # Calculate α, which needs tau
    alpha = np.gradient(tau) / np.gradient(q)

    # Calculate Dq, which needs tau and q
    f = q * alpha - tau

    if show is True:
        singularity_spectrum_plot(alpha, f)

    return alpha, f

# Scaling exponents


def scaling_exponents(signal, q="default", lim=[None, None],
                      windows="default", overlap=True, integrate=True,
                      order=1, show=False):
    """Calculate the multifractal scaling exponents `τ`, which is given
    by

    .. math::

       \tau(q) = qh(q) - 1.

    To evaluate the scaling exponent `τ`, plot it vs `q`. If the
    relation between `τ` is linear, the data is monofractal. If not,
    it is multifractal.

    Note that these measures rarely match the theoretical expectation,
    thus a variation of ± 0.25 is absolutely reasonable.

    The parameters are mostly identical to `fractal_mfdfa()`, as one needs to
    perform MFDFA to obtain the singularity spectrum. Calculating only the
    DFA is insufficient, as it only has `q=2`, and a set of `q` values are
    needed. Here defaulted to `q = np.linspace(-10,10,41)`, where the `0`
    element is removed by `_cleanse_q()`.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    q : list or np.array (default `np.linspace(-10,10,41)`)
        The sequence of fractal exponents. Must be a sequence between -10
        and 10 (note that zero will be removed, since the code does not converge
        there). If "default", will takes the form `np.linspace(-10,10,41)`.

    lim: list (default `[1, lag.size//2]`)
        List of lower and upper lag limits. If none, the polynomial fittings
        will be restrict to half the maximal lag and discard the first lag
        point.

    windows : list
        A list containing the lengths of the windows (number of data points in
        each subseries). Also referred to as 'lag' or 'scale'. If 'default',
        will set it to a logarithmic scale (so that each window scale has the
        same weight) with a minimum of 4 and maximum of a tenth of the length
        (to have more than 10 windows to calculate the average fluctuation).

    overlap : bool
        Defaults to True, where the windows will have a 50% overlap
        with each other, otherwise non-overlapping windows will be used.

    integrate : bool
        It is common practice to convert the signal to a random walk (i.e.,
        detrend and integrate, which corresponds to the signal 'profile'). Note
        that it leads to the flattening of the signal, which can lead to the
        loss of some details (see Ihlen, 2012 for an explanation). Note that for
        strongly anticorrelated signals, this transformation should be applied
        two times (i.e., provide `np.cumsum(signal - np.mean(signal))` instead
        of `signal`).

    order : int
        The order of the polynomial detrending, 1 for the linear trend.

    show : bool
        Visualise the multifractal scaling exponents.

    Returns
    -------
    q: np.array
        The `q` powers.

    tau: np.array
        Scaling exponents `τ`. A usually increasing function of `q` from
        which the fractality of the data can be determined by its shape. A truly
        linear tau indicates monofractality, whereas a curved one (usually
        curving around small `q` values) indicates multifractality.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=3, noise=0.05)
    >>> q, tau = nk.scaling_exponents(signal, show=True)
    >>> q, tau #doctest: +SKIP


    References
    -----------
    - Ihlen, E. A. F. E. (2012). Introduction to multifractal detrended
      fluctuation analysis in Matlab. Frontiers in physiology, 3, 141.

    - J. W. Kantelhardt, S. A. Zschiegner, E. Koscielny-Bunde, S. Havlin, A.
      Bunde, H. E. Stanley (2002). Multifractal detrended fluctuation analysis
      of nonstationary time series. Physica A, 316(1-4), 87–114.

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """

    # Draw q values
    if isinstance(q, str):
        q = list(np.linspace(-10, 10, 41))

    q = _cleanse_q(q)

    # obtain the windows and fluctuations
    windows, fluctuations = _fractal_dfa(signal=signal,
                                         windows=windows,
                                         overlap=overlap,
                                         integrate=integrate,
                                         order=order,
                                         multifractal=True,
                                         q=q
                                         )

    # flatten q for the multiplications ahead
    q = q.flatten()

    # if no limits given
    if lim[0] is None and lim[1] is None:
        lim = [1, windows.size // 2]

    # Calculate the slopes
    slopes = _slopes(windows, fluctuations, q, lim)

    # Calculate τ
    tau = q * slopes - 1

    if show is True:
        scaling_exponents_plot(q, tau)

    return q, tau

# Generalised Hurst exponents


def hurst_exponents(signal, q="default", lim=[None, None], windows="default",
                    overlap=True, integrate=True, order=1, show=False):
    """Calculate the generalised Hurst exponents `h(q)` from the multifractal
    `fractal_dfa()`, which are simply the slopes of each DFA for various `q`
    powers.

    Note that these measures rarely match the theoretical expectation,
    thus a variation of ± 0.25 is absolutely reasonable.

    The parameters are mostly identical to `fractal_mfdfa()`, as one needs to
    perform MFDFA to obtain the singularity spectrum. Calculating only the
    DFA is insufficient, as it only has `q=2`, and a set of `q` values are
    needed. Here defaulted to `q = np.linspace(-10,10,41)`, where the `0`
    element is removed by `_cleanse_q()`.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    q : list or np.array (default `np.linspace(-10,10,41)`)
        The sequence of fractal exponents. Must be a sequence between -10
        and 10 (note that zero will be removed, since the code does not converge
        there). If "default", will takes the form `np.linspace(-10,10,41)`.

    lim: list (default `[1, lag.size//2]`)
        List of lower and upper lag limits. If none, the polynomial fittings
        will be restrict to half the maximal lag and discard the first lag
        point.

    windows : list
        A list containing the lengths of the windows (number of data points in
        each subseries). Also referred to as 'lag' or 'scale'. If 'default',
        will set it to a logarithmic scale (so that each window scale has the
        same weight) with a minimum of 4 and maximum of a tenth of the length
        (to have more than 10 windows to calculate the average fluctuation).

    overlap : bool
        Defaults to True, where the windows will have a 50% overlap
        with each other, otherwise non-overlapping windows will be used.

    integrate : bool
        It is common practice to convert the signal to a random walk (i.e.,
        detrend and integrate, which corresponds to the signal 'profile'). Note
        that it leads to the flattening of the signal, which can lead to the
        loss of some details (see Ihlen, 2012 for an explanation). Note that for
        strongly anticorrelated signals, this transformation should be applied
        two times (i.e., provide `np.cumsum(signal - np.mean(signal))` instead
        of `signal`).

    order : int
        The order of the polynomial detrending, 1 for the linear trend.

    show : bool
        Visualise the singularity plot.

    Returns
    -------
    q: np.array
        The `q` powers.

    hq: np.array
        Singularity strength `h(q)`. The width of this function indicates the
        strength of the multifractality. A width of `max(h(q)) - min(h(q)) ≈ 0`
        means the data is monofractal.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=3, noise=0.05)
    >>> q, hq = nk.hurst_exponents(signal, show=True)
    >>> q, hq #doctest: +SKIP


    References
    -----------
    - Ihlen, E. A. F. E. (2012). Introduction to multifractal detrended
      fluctuation analysis in Matlab. Frontiers in physiology, 3, 141.

    - J. W. Kantelhardt, S. A. Zschiegner, E. Koscielny-Bunde, S. Havlin, A.
      Bunde, H. E. Stanley (2002). Multifractal detrended fluctuation analysis
      of nonstationary time series. Physica A, 316(1-4), 87–114.

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """

    # Draw q values
    if isinstance(q, str):
        q = list(np.linspace(-10, 10, 41))

    q = _cleanse_q(q)

    # obtain the windows and fluctuations
    windows, fluctuations = _fractal_dfa(signal=signal,
                                         windows=windows,
                                         overlap=overlap,
                                         integrate=integrate,
                                         order=order,
                                         multifractal=True,
                                         q=q
                                         )

    # flatten q for the multiplications ahead
    q = q.flatten()

    # if no limits given
    if lim[0] is None and lim[1] is None:
        lim = [1, windows.size // 2]

    # Calculate the slopes
    hq = _slopes(windows, fluctuations, q, lim)

    if show is True:
        hurst_exponents_plot(q, hq)

    return q, hq


def _cleanse_q(q=2):
    # TODO: Add log calculator for q ≈ 0

    # Fractal powers as floats
    q = np.asarray_chkfinite(q, dtype=np.float)

    # Ensure q≈0 is removed, since it does not converge. Limit set at |q| < 0.1
    q = q[(q < -0.1) + (q > 0.1)]

    # Reshape q to perform np.float_power
    q = q.reshape(-1, 1)

    return q


def _slopes(windows, fluctuations, q, lim=[None, None]):
    """
    Extra the slopes of each `q` power obtained with MFDFA to later produce
    either the singularity spectrum or the multifractal exponents.

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """

    # if no limits given
    if lim[0] is None and lim[1] is None:
        lim = [windows[1], windows[windows.size // 2]]

    # clean q
    q = _cleanse_q(q)

    # Fractal powers as floats
    q = np.asarray_chkfinite(q, dtype=float)

    # Ensure mfdfa has the same q-power entries as q
    if fluctuations.shape[1] != q.shape[0]:
        raise ValueError(
            "Fluctuation function and q powers don't match in dimension.")

    # Allocated array for slopes
    dfa = np.zeros(len(q))

    # Find slopes of each q-power
    for i in range(len(q)):
        dfa[i] = np.polyfit(np.log2(windows), np.log2(fluctuations[:, i]), 1)[0]

    return dfa


def singularity_spectrum_plot(alpha, f):
    """
    Plots the singularity spectrum.

    Parameters
    ----------
    hq: np.array
        Singularity strength `hq` as calculated with `singularity_spectrum()`.

    Dq: np.array
        Singularity spectrum `Dq` as calculated with `singularity_spectrum()`.

    Returns
    -------
    fig: matplotlib fig
        Returns the figure, useful if one wishes to use fig.savefig(...).

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """

    plt.plot(alpha, f, 'o-')

    plt.title("Singularity Spectrum")
    plt.ylabel(r'f(α)')
    plt.xlabel(r'α')
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

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """

    plt.plot(q, tau, 'o-')

    plt.title("Scaling Exponents")
    plt.ylabel(r'τ(q)')
    plt.xlabel(r'q')
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

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """

    plt.plot(q, hq, 'o-')

    plt.title("Generalised Hurst Exponents")
    plt.ylabel(r'h(q)')
    plt.xlabel(r'q')
    # plt.legend()
    plt.show()

    return None
