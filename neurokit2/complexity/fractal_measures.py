# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from .fractal_dfa import _fractal_dfa, _cleanse_q

## This is based on Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E.,
# Havlin, S., Bunde, A., & Stanley, H. E., Multifractal detrended fluctuation
# analysis of nonstationary time series. Physica A, 316(1-4), 87-114, 2002 as
# well as on nolds (https://github.com/CSchoel/nolds) and on work by  Espen A.
# F. Ihlen, Introduction to multifractal detrended fluctuation analysis in
# Matlab, Front. Physiol., 2012, https://doi.org/10.3389/fphys.2012.00141
#
# It was designed by Leonardo Rydin Gorjão as part of MFDFA
# (https://github.com/LRydin/MFDFA). It is included here by the author and
# altered to fit NK to the best of its extent.

def singularity_spectrum(signal, q = "default", lim = [None, None],
    windows = "default", overlap = True,  integrate = True, order = 1,
    show = False):
    """Extract the slopes of the fluctuation function to futher obtain the
    singularity strength `α` (or Hölder exponents) and singularity spectrum
    `f(α)` (or fractal dimension). This is iconically shaped as an inverse
    parabola, but most often it is difficult to obtain the negative `q` terms,
    and one can focus on the left side of the parabola (`q>0`).

    The parameters are mostly identical to `fractal_mfdfa()`, as one needs to
    perform MFDFA to obtain the singularity spectrum. Calculating only the
    DFA is insufficient, as it only has `q=2`, and a set of `q` values are
    needed. Here defaulted to `q = list(range(-5,5))`, where the `0` element
    is removed by `_cleanse_q()`.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    q : list
        The sequence of fractal exponents. Must be a sequence between -10
        and 10 (note that zero will be removed, since the code does not converge
        there). If "default", will become 'np.linspace(-5,5,41)'.
    lim: list (default `[1, lag.size//2]`)
        List of lower and upper lag limits. If none, the polynomial fittings
        will be restrict to half the maximal lag and discard the first lag
        point.
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

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """

    # Draw q values
    if isinstance(q, str):
        q = list(np.linspace(-10,10,41))
    q = _cleanse_q(q)

    # obtain the windows and fluctuations
    windows, fluctuations = _fractal_dfa(
                                signal = signal,
                                windows = windows,
                                overlap = overlap,
                                integrate = integrate,
                                order = order,
                                multifractal = True,
                                q = q
                            )

    # flatten q for the multiplications ahead
    q = q.flatten()

    alpha, f = _singularity_spectrum(
                    lag = windows,
                    mfdfa = fluctuations,
                    q = q,
                    lim = lim
                )

    if show is True:
        singularity_spectrum_plot(alpha, f)

    return alpha, f

def _singularity_spectrum(lag, mfdfa, q, lim = [None, None]):
    """Extract the slopes of the fluctuation function to obtain the singularity
    strength `α` (or Hölder exponents) and singularity spectrum `f(α)` (or
    fractal dimension). This is iconically shaped as an inverse parabola, but
    most often it is difficult to obtain the negative `q` terms, and one can
    focus on the left side of the parabola (`q>0`).

    Parameters
    ----------
    lag: np.array of ints
        An array with the window sizes which where used in MFDFA.

    mdfda: np.ndarray
        Matrix of the fluctuation function from MFDFA

    q: np.array
        Fractal exponents used. Must be more than 2 points.

    lim: list (default `[1, lag.size//2]`)
        List of lower and upper lag limits. If none, the polynomial fittings
        will be restrict to half the maximal lag and discard the first lag
        point.

    Returns
    -------
    alpha: np.array
        Singularity strength `α`. The width of this function indicates the
        strength of the multifractality. A width of `max(α) - min(α) ≈ 0`
        means the data is monofractal.

    f: np.array
        Singularity spectrum `f(α)`. The location of the maximum of `f(α)`
        (with `α` as the abscissa) should be 1 and indicates the most
        prominent fractal scale in the data.

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """

    # Calculate tau
    _, tau = _scaling_exponents(lag, mfdfa, q = q, lim = lim)

    # Calculate α, which needs tau
    alpha = ( np.gradient(tau) / np.gradient(q) )

    # Calculate Dq, which needs tau and q
    f = _falpha(tau, alpha, q)

    return alpha, f

##### Scaling exponents

def scaling_exponents(signal, q = list(range(-10,10)), lim = [None, None],
    windows = "default", overlap = True,  integrate = True, order = 1,
    show = False):
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
    needed. Here defaulted to `q = list(range(-5,5))`, where the `0` element
    is removed by `_cleanse_q()`.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    q : list (default 'list(range(-10,10))')
        The sequence of fractal exponents. Must be a sequence between -10
        and 10 (note that zero will be removed, since the code does not converge there).
    lim: list (default `[1, lag.size//2]`)
        List of lower and upper lag limits. If none, the polynomial fittings
        will be restrict to half the maximal lag and discard the first lag
        point.
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

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """

    # Draw q values
    if isinstance(q, str):
        q = list(np.linspace(-10,10,41))

    q = _cleanse_q(q)

    # obtain the windows and fluctuations
    windows, fluctuations = _fractal_dfa(
                                signal = signal,
                                windows = windows,
                                overlap = overlap,
                                integrate = integrate,
                                order = order,
                                multifractal = True,
                                q = q
                            )


    # flatten q for the multiplications ahead
    q = q.flatten()

    q, tau = _scaling_exponents(
                lag = windows,
                mfdfa = fluctuations,
                q = q,
                lim = lim
                )

    if show is True:
        scaling_exponents_plot(q, tau)

    return q, tau

def _scaling_exponents(lag, mfdfa, q, lim = [None, None]):
    """Calculate the multifractal scaling exponents `τ`, which is given
    by

    .. math::

       \tau(q) = qh(q) - 1.

    To evaluate the scaling exponent `τ`, plot it vs `q`. If the
    relation between `τ` is linear, the data is monofractal. If not,
    it is multifractal.
    Note that these measures rarely match the theoretical expectation,
    thus a variation of ± 0.25 is absolutely reasonable.


    Parameters
    ----------
    lag: np.array of ints
        An array with the window sizes which where used in MFDFA.

    mdfda: np.ndarray
        Matrix of the fluctuation function from MFDFA

    q: np.array
        Fractal exponents used. Must be more than 2 points.

    lim: list (default `[1, lag.size//2]`)
        List of lower and upper lag limits. If none, the polynomial fittings
        will be restrict to half the maximal lag and discard the first lag
        point.

    Returns
    -------
    q: np.array
        The `q` powers.

    tau: np.array
        Scaling exponents `τ`. A usually increasing function of `q` from
        which the fractality of the data can be determined by its shape. A truly
        linear tau indicates monofractality, whereas a curved one (usually
        curving around small `q` values) indicates multifractality.


    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """

    # Calculate the slopes
    slopes = _slopes(lag, mfdfa, q, lim)

    return q, ( q * slopes ) - 1


##### Generalised Hurst exponents

def hurst_exponents(signal, q = "default", lim = [None, None],
    windows = "default", overlap = True,  integrate = True, order = 1,
    show = False):
    """Calculate the generalised Hurst exponents `h(q)` from the multifractal
    `fractal_dfa()`, which are simply the slopes of each DFA for various `q`
    powers.

    The parameters are mostly identical to `fractal_mfdfa()`, as one needs to
    perform MFDFA to obtain the singularity spectrum. Calculating only the
    DFA is insufficient, as it only has `q=2`, and a set of `q` values are
    needed. Here defaulted to `q = list(range(-5,5))`, where the `0` element
    is removed by `_cleanse_q()`.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    q : list
        The sequence of fractal exponents. Must be a sequence between -10
        and 10 (note that zero will be removed, since the code does not converge
        there). If "default", will become 'np.linspace(-5,5,41)'.
    lim: list (default `[1, lag.size//2]`)
        List of lower and upper lag limits. If none, the polynomial fittings
        will be restrict to half the maximal lag and discard the first lag
        point.
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

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """

    # Draw q values
    if isinstance(q, str):
        q = list(np.linspace(-10,10,41))
    q = _cleanse_q(q)

    # obtain the windows and fluctuations
    windows, fluctuations = _fractal_dfa(
                                signal = signal,
                                windows = windows,
                                overlap = overlap,
                                integrate = integrate,
                                order = order,
                                multifractal = True,
                                q = q
                            )

    # flatten q for the multiplications ahead
    q = q.flatten()

    q, hq = _hurst_exponents(
                lag = windows,
                mfdfa = fluctuations,
                q = q,
                lim = lim
                )

    if show is True:
        hurst_exponents_plot(q, tau)

    return q, hq


def _hurst_exponents(lag, mfdfa, q, lim = [None, None]):
    """Calculate the generalised Hurst exponents `h(q)` from the multifractal
    `fractal_dfa()`, which are simply the slopes of each DFA for various `q`
    powers.

    Note that these measures rarely match the theoretical expectation,
    thus a variation of ± 0.25 is absolutely reasonable.

    Parameters
    ----------
    lag: np.array of ints
        An array with the window sizes which where used in MFDFA.

    mdfda: np.ndarray
        Matrix of the fluctuation function from MFDFA

    q: np.array
        Fractal exponents used. Must be more than 2 points.

    lim: list (default `[1, lag.size//2]`)
        List of lower and upper lag limits. If none, the polynomial fittings
        will be restrict to half the maximal lag and discard the first lag
        point.

    Returns
    -------
    q: np.array
        The `q` powers.

    hq: np.array
        Singularity strength `h(q)`. The width of this function indicates the
        strength of the multifractality. A width of `max(h(q)) - min(h(q)) ≈ 0`
        means the data is monofractal.

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """

    # if no limits given
    if lim[0] is None and lim[1] is None:
        lim = [1, lag.size//2]

    # clean q
    q = _cleanse_q(q)

    # Calculate the slopes
    hq = _slopes(lag, mfdfa, q, lim)

    return q, hq


def _slopes(lag, mfdfa, q, lim = [None, None]):
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
        lim = [lag[1], lag[lag.size//2]]

    # clean q
    q = _cleanse_q(q)

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
        slopes[i] = np.polynomial.polynomial.polyfit(
                        np.log(lag[lim[0]:lim[1]]),
                        np.log(mfdfa[lim[0]:lim[1],i]),
                        1
                    )[1]

    return slopes

def _falpha(tau, alpha, q):
    """
    Calculate the singularity spectrum or fractal dimension `f(α)`.

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """
    return q * alpha - tau

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
