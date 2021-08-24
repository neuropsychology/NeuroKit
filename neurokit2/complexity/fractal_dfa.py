# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..misc import NeuroKitWarning, expspace


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

       The order of the polynomial trend for detrending, 1 for the linear trend.

    multifractal : bool
        If true, compute Multifractal Detrended Fluctuation Analysis (MFDFA), in
        which case the argument `q` is taken into account.

    q : list or np.array (default `2`)
        The sequence of fractal exponents when `multifractal=True`. Must be a
        sequence between `-10` and `10` (note that zero will be removed, since
        the code does not converge there). Setting `q = 2` (default) gives a
        result of a standard DFA. For instance, Ihlen (2012) uses
        `q = [-5, -3, -1, 0, 1, 3, 5]`. In general, positive q moments amplify
        the contribution of fractal components with larger amplitude and
        negative q moments amplify the contribution of fractal with smaller
        amplitude (Kantelhardt et al., 2002)

    show : bool
        Visualise the trend between the window size and the fluctuations.

    Returns
    ----------
    dfa : dict
        If `multifractal` is False, the dictionary contains q value, a series of windows, fluctuations of each window and the
        slopes value of the log2(windows) versus log2(fluctuations) plot. If `multifractal` is True, the dictionary
        additionally contains the parameters of the singularity spectrum (scaling exponents, singularity dimension, singularity
        strength; see `singularity_spectrum()` for more information).


    Examples
    ----------
    >>> import neurokit2 as nk
    >>> import numpy as np
    >>>
    >>> signal = nk.signal_simulate(duration=10, noise=0.05)
    >>> dfa1 = nk.fractal_dfa(signal, show=True)
    >>> dfa1 #doctest: +SKIP
    >>> dfa2 = nk.fractal_mfdfa(signal, q=np.arange(-5, 6), show=True)
    >>> dfa2 #doctest: +SKIP


    References
    -----------
    - Ihlen, E. A. F. E. (2012). Introduction to multifractal detrended
      fluctuation analysis in Matlab. Frontiers in physiology, 3, 141.

    - Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E., Havlin, S.,
      Bunde, A., & Stanley, H. E. (2002). Multifractal detrended fluctuation
      analysis of nonstationary time series. Physica A: Statistical
      Mechanics and its Applications, 316(1-4), 87-114.

    - Hardstone, R., Poil, S. S., Schiavone, G., Jansen, R., Nikulin, V. V.,
      Mansvelder, H. D., & Linkenkaer-Hansen, K. (2012). Detrended
      fluctuation analysis: a scale-free view on neuronal oscillations.
      Frontiers in physiology, 3, 450.

    - `nolds <https://github.com/CSchoel/nolds/>`_

    - `MFDFA <https://github.com/LRydin/MFDFA/>`_

    - `Youtube introduction <https://www.youtube.com/watch?v=o0LndP2OlUI>`_

    """
    # Sanity checks
    n = len(signal)
    windows = _fractal_dfa_findwindows(n, windows)
    _fractal_dfa_findwindows_warning(windows, n)  # Return warning for too short windows

    # Preprocessing
    if integrate is True:
        signal = np.cumsum(signal - np.mean(signal))  # Get signal profile

    # Sanitize fractal power (cannot be close to 0)
    q = _cleanse_q(q, multifractal=multifractal)

    # obtain the windows and fluctuations
    windows, fluctuations = _fractal_dfa(signal=signal,
                                         windows=windows,
                                         overlap=overlap,
                                         integrate=integrate,
                                         order=order,
                                         multifractal=multifractal,
                                         q=q
                                         )

    if len(fluctuations) == 0:
        return np.nan

    slopes = _slopes(windows, fluctuations, q)
    out = {'q' : q[:, 0],
           'windows' : windows,
           'fluctuations' : fluctuations,
           'slopes' : slopes}

    if multifractal is True:
        singularity = singularity_spectrum(windows=windows,
                                           fluctuations=fluctuations,
                                           q=q,
                                           slopes=slopes)
        out.update(singularity)

    # Plot if show is True.
    if show is True:
        if multifractal is True:
            _fractal_mdfa_plot(windows=windows,
                               fluctuations=fluctuations,
                               multifractal=multifractal,
                               q=q,
                               tau=out['tau'],
                               hq=out['hq'],
                               Dq=out['Dq'])
        else:
            _fractal_dfa_plot(windows=windows,
                              fluctuations=fluctuations,
                              multifractal=multifractal,
                              q=q)

    return out

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

    # Function to store fluctuations. For DFA this is an array. For MFDFA, this
    # is a matrix of size (len(windows),len(q))
    n = len(signal)
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

def singularity_spectrum(windows, fluctuations, q, slopes):
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
    windows : list
        A list containing the lengths of the windows. Output of `_fractal_dfa()`.

    fluctuations : np.ndarray
        The detrended fluctuations, from DFA or MFDFA. Output of `_fractal_dfa()`.

    q : list or np.array (default `np.linspace(-10,10,41)`)
        The sequence of fractal exponents. Must be a sequence between -10
        and 10 (note that zero will be removed, since the code does not converge
        there). If "default", will takes the form `np.linspace(-10,10,41)`.

    slopes : np.ndarray
        The slopes of each `q` power obtained with MFDFA. Output of `_slopes()`.

    Returns
    -------
    tau: np.array
        Scaling exponents `τ`. A usually increasing function of `q` from
        which the fractality of the data can be determined by its shape. A truly
        linear tau indicates monofractality, whereas a curved one (usually
        curving around small `q` values) indicates multifractality.

    hq: np.array
        Singularity strength `hq`. The width of this function indicates the
        strength of the multifractality. A width of `max(hq) - min(hq) ≈ 0`
        means the data is monofractal.

    Dq: np.array
        Singularity spectrum `Dq`. The location of the maximum of `Dq` (with
         `hq` as the abscissa) should be 1 and indicates the most prominent
         exponent in the data.

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.
    """

    # Calculate τ
    tau = q[:, 0] * slopes - 1

    # Calculate hq or α, which needs tau
    hq = np.gradient(tau) / np.gradient(q[:, 0])

    # Calculate Dq or f(α), which needs tau and q
    Dq = q[:, 0] * hq - tau

    # Calculate the singularity
    ExpRange = np.max(hq) - np.min(hq)
    ExpMean = np.mean(hq)
    DimRange = np.max(Dq) - np.min(Dq)
    DimMean = np.mean(Dq)
    out = {'tau': tau,
           'hq': hq,
           'Dq': Dq,
           'ExpRange': ExpRange,
           'ExpMean': ExpMean,
           'DimRange': DimRange,
           'DimMean': DimMean}

    return out


# =============================================================================
#  Utils
# =============================================================================

def _cleanse_q(q=2, multifractal=False):
    # TODO: Add log calculator for q ≈ 0

    # Enforce DFA in case 'multifractal = False' but 'q' is not 2
    if multifractal is False:
        q = 2
    else:
        if isinstance(q, int):
            warn(
                "For multifractal DFA, q needs to be a list. "
                "Using the default value of q = [-5, -3, -1, 0, 1, 3, 5]",
                category=NeuroKitWarning
            )
            q = [-5, -3, -1, 0, 1, 3, 5]

    # Fractal powers as floats
    q = np.asarray_chkfinite(q, dtype=float)

    # Ensure q≈0 is removed, since it does not converge. Limit set at |q| < 0.1
    q = q[(q < -0.1) + (q > 0.1)]

    # Reshape q to perform np.float_power
    q = q.reshape(-1, 1)

    return q


def _slopes(windows, fluctuations, q):
    """
    Extract the slopes of each `q` power obtained with MFDFA to later produce
    either the singularity spectrum or the multifractal exponents.

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """

    # Ensure mfdfa has the same q-power entries as q
    if fluctuations.shape[1] != q.shape[0]:
        raise ValueError(
            "Fluctuation function and q powers don't match in dimension.")

    # Allocated array for slopes
    slopes = np.zeros(len(q))
    # Find slopes of each q-power
    for i in range(len(q)):
        slopes[i] = np.polyfit(np.log2(windows), np.log2(fluctuations[:, i]), 1)[0]

    return slopes


def _fractal_dfa_findwindows(n, windows="default"):
    # Convert to array
    if isinstance(windows, list):
        windows = np.asarray(windows)

    # Default windows number
    if windows is None or isinstance(windows, str):
        windows = int(n / 10)

    # Default windows sequence
    if isinstance(windows, int):
        windows = expspace(
            10, int(n / 10), windows, base=2
        )  # see https://github.com/neuropsychology/NeuroKit/issues/206
        windows = np.unique(windows)  # keep only unique

    return windows


def _fractal_dfa_findwindows_warning(windows, n):

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

    else:
        # Compute Root Mean Square (RMS)
        fluctuation = np.sum(detrended ** 2, axis=1) / detrended.shape[1]
        fluctuation = np.sqrt(np.sum(fluctuation) / len(fluctuation))

    return fluctuation

# =============================================================================
#  Plots
# =============================================================================


def _fractal_mdfa_plot(windows, fluctuations, multifractal, q, tau, hq, Dq):

    # Prepare figure
    fig = plt.figure(constrained_layout=False)
    spec = matplotlib.gridspec.GridSpec(
        ncols=2, nrows=2
    )

    ax_fluctuation = fig.add_subplot(spec[0, 0])
    ax_spectrum = fig.add_subplot(spec[0, 1])
    ax_tau = fig.add_subplot(spec[1, 0])
    ax_hq = fig.add_subplot(spec[1, 1])

    n = len(q)
    colors = plt.cm.plasma(np.linspace(0, 1, n))

    # Plot the fluctuation plot
    # Plot the points
    for i in range(len(q)):
        polyfit = np.polyfit(np.log2(windows), np.log2(fluctuations[:, i]), 1)
        ax_fluctuation.loglog(windows, fluctuations, "bo", c='#d2dade', markersize=5, base=2)
    # Plot the polyfit line
    for i in range(len(q)):
        polyfit = np.polyfit(np.log2(windows), np.log2(fluctuations[:, i]), 1)
        fluctfit = 2 ** np.polyval(polyfit, np.log2(windows))
        ax_fluctuation.loglog(windows, fluctfit, "r", c=colors[i], base=2, label='_no_legend_')
    ax_fluctuation.plot([],
                        label=(r"$\alpha$ = {:.3f}, q={:.1f}").format(polyfit[0], q[0][0]),
                        c=colors[0])
    ax_fluctuation.plot([],
                        label=(r"$\alpha$ = {:.3f}, q={:.1f}").format(polyfit[0], q[-1][0]),
                        c=colors[-1])

    # Plot the singularity spectrum
    _singularity_spectrum_plot(hq, Dq, ax=ax_spectrum)
    # Plot tau against q
    _scaling_exponents_plot(q, tau, ax=ax_tau)
    # Plot hq against q
    _hurst_exponents_plot(q, hq, ax=ax_hq)

    ax_fluctuation.set_xlabel(r"$\log_{2}$(Window)")
    ax_fluctuation.set_ylabel(r"$\log_{2}$(Fluctuation)")
    ax_fluctuation.legend(loc="lower right")
    fig.suptitle('Multifractal Detrended Fluctuation Analysis')

    return None


def _fractal_dfa_plot(windows, fluctuations, multifractal, q):

    polyfit = np.polyfit(np.log2(windows), np.log2(fluctuations), 1)
    fluctfit = 2 ** np.polyval(polyfit, np.log2(windows))
    plt.loglog(windows, fluctuations, "bo", c='#90A4AE')
    plt.loglog(windows, fluctfit, "r", c='#E91E63',
               label=r"$\alpha$ = {:.3f}".format(polyfit[0][0]))
    plt.legend(loc="lower right")
    plt.title('Detrended Fluctuation Analysis')

    return None


def _singularity_spectrum_plot(hq, Dq, ax=None):
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

    ax.plot(hq, Dq, 'o-', c='#FFC107')

#    ax.set_title("Singularity Spectrum")
    ax.set_ylabel(r'Singularity dimension ($D_q$)')
    ax.set_xlabel(r'Singularity exponent ($h_q$)')

    return None


def _scaling_exponents_plot(q, tau, ax=None):
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

    ax.plot(q, tau, 'o-', c='#E91E63')

#    ax.set_title("Scaling Exponents")
    ax.set_ylabel(r'Scaling exponents ($τ_q$)')
    ax.set_xlabel(r'Multifractal parameter ($q$)')

    return None


def _hurst_exponents_plot(q, hq, ax=None):
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

    ax.plot(q, hq, 'o-', c='#2196F3')

#    ax.set_title("Generalised Hurst Exponents")
    ax.set_ylabel(r'Generalized Hurst Exponents ($h_q$)')
    ax.set_xlabel(r'Multifractal parameter ($q$)')

    return None
