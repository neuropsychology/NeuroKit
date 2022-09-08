# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import find_knee


def fractal_dfa(
    signal,
    scale="default",
    overlap=True,
    integrate=True,
    order=1,
    multifractal=False,
    q="default",
    maxdfa=False,
    show=False,
    **kwargs,
):
    """**(Multifractal) Detrended Fluctuation Analysis (DFA or MFDFA)**

    Detrended fluctuation analysis (DFA) is used to find long-term statistical dependencies in time
    series.

    For monofractal DFA, the output *alpha* :math:`\\alpha` corresponds to the slope of the linear
    trend between the scale factors and the fluctuations. For multifractal DFA, the slope values
    under different *q* values are actually generalised Hurst exponents *h*. Monofractal DFA
    corresponds to MFDFA with *q = 2*, and its output is actually an estimation of the
    **Hurst exponent** (:math:`h_{(2)}`).

    The Hurst exponent is the measure of long range autocorrelation of a signal, and
    :math:`h_{(2)} > 0.5` suggests the presence of long range correlation, while
    :math:`h_{(2)} < 0.5`suggests short range correlations. If :math:`h_{(2)} = 0.5`, it indicates
    uncorrelated indiscriminative fluctuations, i.e. a Brownian motion.

    .. figure:: ../img/douglas2022a.png
       :alt: Illustration of DFA (Douglas et al., 2022).

    Multifractal DFA returns the generalised Hurst exponents *h* for different values of *q*. It is
    converted to the multifractal **scaling exponent** *Tau* :math:`\\tau_{(q)}`, which non-linear
    relationship with *q* can indicate multifractility. From there, we derive the singularity
    exponent *H* (or :math:`\\alpha`) (also known as Hölder's exponents) and the singularity
    dimension *D* (or :math:`f(\\alpha)`). The variation of *D* with *H* is known as multifractal
    singularity spectrum (MSP), and usually has shape of an inverted parabola. It measures the long
    range correlation property of a signal. From these elements, different features are extracted:

    * **Width**: The width of the singularity spectrum, which quantifies the degree of the
      multifractality. In the case of monofractal signals, the MSP width is zero, since *h*\\(q) is
      independent of *q*.
    * **Peak**: The value of the singularity exponent *H* corresponding to the peak of
      singularity dimension *D*. It is a measure of the self-affinity of the signal, and a high
      value is an indicator of high degree of correlation between the data points. In the other
      words, the process is recurrent and repetitive.
    * **Mean**: The mean of the maximum and minimum values of singularity exponent *H*, which
      quantifies the average fluctuations of the signal.
    * **Max**: The value of singularity spectrum *D* corresponding to the maximum value of
      singularity exponent *H*, which indicates the maximum fluctuation of the signal.
    * **Delta**: the vertical distance between the singularity spectrum *D* where the singularity
      exponents are at their minimum and maximum. Corresponds to the range of fluctuations of the
      signal.
    * **Asymmetry**: The Asymmetric Ratio (AR) corresponds to the centrality of the peak of the
      spectrum. AR = 0.5 indicates that the multifractal spectrum is symmetric (Orozco-Duque et
      al., 2015).
    * **Fluctuation**: The *h*-fluctuation index (hFI) is defined as the power of the second
      derivative of *h*\\(q). See Orozco-Duque et al. (2015).
    * **Increment**: The cumulative function of the squared increments (:math:`\\alpha CF`) of the
      generalized Hurst's exponents between consecutive moment orders is a more robust index of the
      distribution of the generalized Hurst's exponents (Faini et al., 2021).

    This function can be called either via ``fractal_dfa()`` or ``complexity_dfa()``, and its
    multifractal variant can be directly accessed via ``fractal_mfdfa()`` or ``complexity_mfdfa()``.

    .. note ::

      Help is needed to implement the modified formula to compute the slope when
      *q* = 0. See for instance Faini et al. (2021). See https://github.com/LRydin/MFDFA/issues/33

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    scale : list
        A list containing the lengths of the windows (number of data points in each subseries) that
        the signal is divided into. Also referred to as Tau :math:`\\tau`. If ``"default"``, will
        set it to a logarithmic scale (so that each window scale has the same weight) with a
        minimum of 4 and maximum of a tenth of the length (to have more than 10 windows to
        calculate the average fluctuation).
    overlap : bool
        Defaults to ``True``, where the windows will have a 50% overlap with each other, otherwise
        non-overlapping windows will be used.
    integrate : bool
        It is common practice to convert the signal to a random walk (i.e., detrend and integrate,
        which corresponds to the signal 'profile') in order to avoid having too small exponent
        values. Note that it leads to the flattening of the signal, which can lead to the loss of
        some details (see Ihlen, 2012 for an explanation). Note that for strongly anticorrelated
        signals, this transformation should be applied  two times (i.e., provide
        ``np.cumsum(signal - np.mean(signal))`` instead of ``signal``).
    order : int
       The order of the polynomial trend for detrending. 1 corresponds to a linear detrending.
    multifractal : bool
        If ``True``, compute Multifractal Detrended Fluctuation Analysis (MFDFA), in which case the
        argument ``q`` is taken into account.
    q : Union[int, list, np.array]
        The sequence of fractal exponents when ``multifractal=True``. Must be a sequence between
        -10 and 10 (note that zero will be removed, since the code does not converge there).
        Setting ``q = 2`` (default for DFA) gives a result of a standard DFA. For instance, Ihlen
        (2012) uses ``q = [-5, -3, -1, 0, 1, 3, 5]`` (default when for multifractal). In general,
        positive q moments amplify the contribution of fractal components with larger amplitude and
        negative q moments amplify the contribution of fractal with smaller amplitude (Kantelhardt
        et al., 2002).
    maxdfa : bool
        If ``True``, it will locate the knee of the fluctuations (using :func:`.find_knee`) and use
        that as a maximum scale value. It computes max. DFA (a similar method exists in
        :func:`entropy_rate`).
    show : bool
        Visualise the trend between the window size and the fluctuations.
    **kwargs : optional
        Currently not used.

    Returns
    ----------
    dfa : float or pd.DataFrame
        If ``multifractal`` is ``False``, one DFA value is returned for a single time series.
    parameters : dict
        A dictionary containing additional information regarding the parameters used
        to compute DFA. If ``multifractal`` is ``False``, the dictionary contains q value, a
        series of windows, fluctuations of each window and the
        slopes value of the log2(windows) versus log2(fluctuations) plot. If
        ``multifractal`` is ``True``, the dictionary additionally contains the
        parameters of the singularity spectrum.

    See Also
    --------
    fractal_hurst, fractal_tmf, entropy_rate

    Examples
    ----------
    **Example 1:** Monofractal DFA

    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=10, frequency=[5, 7, 10, 14], noise=0.05)

      @savefig p_fractal_dfa1.png scale=100%
      dfa, info = nk.fractal_dfa(signal, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      dfa

    As we can see from the plot, the final value, corresponding to the slope of the red line,
    doesn't capture properly the relationship. We can adjust the *scale factors* to capture the
    fractality of short-term fluctuations.

    .. ipython:: python

      scale = nk.expspace(10, 100, 20, base=2)

      @savefig p_fractal_dfa2.png scale=100%
      dfa, info = nk.fractal_dfa(signal, scale=scale, show=True)
      @suppress
      plt.close()

    **Example 2:** Multifractal DFA (MFDFA)

    .. ipython:: python

      @savefig p_fractal_dfa3.png scale=100%
      mfdfa, info = nk.fractal_mfdfa(signal, q=[-5, -3, -1, 0, 1, 3, 5], show=True)
      @suppress
      plt.close()

    .. ipython:: python

      mfdfa

    References
    -----------
    * Faini, A., Parati, G., & Castiglioni, P. (2021). Multiscale assessment of the degree of
      multifractality for physiological time series. Philosophical Transactions of the Royal
      Society A, 379(2212), 20200254.
    * Orozco-Duque, A., Novak, D., Kremen, V., & Bustamante, J. (2015). Multifractal analysis for
      grading complex fractionated electrograms in atrial fibrillation. Physiological Measurement,
      36(11), 2269-2284.
    * Ihlen, E. A. F. E. (2012). Introduction to multifractal detrended
      fluctuation analysis in Matlab. Frontiers in physiology, 3, 141.
    * Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E., Havlin, S.,
      Bunde, A., & Stanley, H. E. (2002). Multifractal detrended fluctuation
      analysis of nonstationary time series. Physica A: Statistical
      Mechanics and its Applications, 316(1-4), 87-114.
    * Hardstone, R., Poil, S. S., Schiavone, G., Jansen, R., Nikulin, V. V.,
      Mansvelder, H. D., & Linkenkaer-Hansen, K. (2012). Detrended
      fluctuation analysis: a scale-free view on neuronal oscillations.
      Frontiers in physiology, 3, 450.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    n = len(signal)
    scale = _fractal_dfa_findscales(n, scale)

    # Sanitize fractal power (cannot be close to 0)
    q = _sanitize_q(q, multifractal=multifractal)

    # Store parameters
    info = {"scale": scale, "q": q}

    # Preprocessing
    if integrate is True:
        # Get signal profile
        signal = np.cumsum(signal - np.mean(signal))

    # Function to store fluctuations. For DFA this is an array. For MFDFA, this
    # is a matrix of size (len(scale),len(q))
    fluctuations = np.zeros((len(scale), len(q)))

    # Start looping over scale
    for i, window in enumerate(scale):

        # Get window
        segments = _fractal_dfa_getwindow(signal, n, window, overlap=overlap)

        # Get polynomial trends
        trends = _fractal_dfa_trends(segments, window, order=order)

        # Get local fluctuation
        fluctuations[i] = _fractal_dfa_fluctuation(segments, trends, q)

    if len(fluctuations) == 0:
        return np.nan, info

    # Max. DFA ---------------------
    if maxdfa is True:
        # Find knees of fluctuations
        knee = np.repeat(len(scale), fluctuations.shape[1])
        for i in range(fluctuations.shape[1]):
            knee[i] = find_knee(
                y=np.log2(fluctuations[:, i]), x=np.log2(scale), show=False, verbose=False
            )
        knee = np.exp2(np.nanmax(knee))
        # Cut fluctuations
        fluctuations = fluctuations[scale <= knee, :]
        scale = scale[scale <= knee]
    # ------------------------------

    # Get slopes
    slopes = _slopes(scale, fluctuations, q)
    if len(slopes) == 1:
        slopes = slopes[0]

    # Prepare output
    info["Fluctuations"] = fluctuations
    info["Alpha"] = slopes

    # Extract features
    if multifractal is True:
        info.update(_singularity_spectrum(q=q, slopes=slopes))
        out = pd.DataFrame(
            {
                k: v
                for k, v in info.items()
                if k
                in [
                    "Peak",
                    "Width",
                    "Mean",
                    "Max",
                    "Delta",
                    "Asymmetry",
                    "Fluctuation",
                    "Increment",
                ]
            },
            index=[0],
        )
    else:
        out = slopes

    # Plot if show is True.
    if show is True:
        if multifractal is True:
            _fractal_mdfa_plot(
                info,
                scale=scale,
                fluctuations=fluctuations,
                q=q,
            )
        else:
            _fractal_dfa_plot(info=info, scale=scale, fluctuations=fluctuations)

    return out, info


# =============================================================================
#  Utils
# =============================================================================
def _fractal_dfa_findscales(n, scale="default"):
    # Convert to array
    if isinstance(scale, list):
        scale = np.asarray(scale)

    # Default scale number
    if scale is None or isinstance(scale, str):
        scale = int(n / 10)

    # See https://github.com/neuropsychology/NeuroKit/issues/206
    if isinstance(scale, int):
        scale = np.exp(np.linspace(np.log(10), np.log(int(n / 10)), scale)).astype(int)
        scale = np.unique(scale)  # keep only unique

    # Sanity checks (return warning for too short scale)
    if len(scale) < 2:
        raise ValueError("NeuroKit error: more than one window is needed. Increase 'scale'.")

    if np.min(scale) < 2:
        raise ValueError(
            "NeuroKit error: there must be at least 2 data points in each window. Decrease 'scale'."
        )
    if np.max(scale) >= n:
        raise ValueError(
            "NeuroKit error: the window cannot contain more data points than the time series. Decrease 'scale'."
        )

    return scale


def _sanitize_q(q=2, multifractal=False):
    # Turn to list
    if isinstance(q, str):
        if multifractal is False:
            q = [2]
        else:
            q = [-5, -3, -1, 0, 1, 3, 5]
    elif isinstance(q, (int, float)):
        q = [q]

    # Fractal powers as floats
    q = np.asarray_chkfinite(q, dtype=float)

    return q


def _slopes(scale, fluctuations, q):
    # Extract the slopes of each `q` power obtained with MFDFA to later produce
    # either the singularity spectrum or the multifractal exponents.
    # Note: added by Leonardo Rydin (see https://github.com/LRydin/MFDFA/)

    # Ensure mfdfa has the same q-power entries as q
    if fluctuations.shape[1] != q.shape[0]:
        raise ValueError("Fluctuation function and q powers don't match in dimension.")

    # Allocated array for slopes
    slopes = np.zeros(len(q))
    # Find slopes of each q-power
    for i in range(len(q)):
        # if fluctiations is zero, log2 wil encounter zero division
        old_setting = np.seterr(divide="ignore", invalid="ignore")
        slopes[i] = np.polyfit(np.log2(scale), np.log2(fluctuations[:, i]), 1)[0]
        np.seterr(**old_setting)

    return slopes


def _fractal_dfa_getwindow(signal, n, window, overlap=True):
    # This function reshapes the segments from a one-dimensional array to a
    # matrix for faster polynomail fittings. 'Overlap' reshapes into several
    # overlapping partitions of the data

    # TODO: see whether this step could be integrated within complexity_coarsegraining

    if overlap:
        segments = np.array([signal[i : i + window] for i in np.arange(0, n - window, window // 2)])
    else:
        segments = signal[: n - (n % window)]
        segments = segments.reshape((signal.shape[0] // window, window))

    return segments


def _fractal_dfa_trends(segments, window, order=1):
    # TODO: can we rely on signal_detrend?
    x = np.arange(window)

    coefs = np.polyfit(x[:window], segments.T, order).T

    # TODO: Could this be optimized? Something like np.polyval(x[:window], coefs)
    trends = np.array([np.polyval(coefs[j], x) for j in np.arange(len(segments))])

    return trends


def _fractal_dfa_fluctuation(segments, trends, q=2):

    # Detrend
    detrended = segments - trends

    # Compute variance
    var = np.var(detrended, axis=1)

    # Remove where var is zero
    var = var[var > 1e-08]
    if len(var) == 0:
        warn("All detrended segments have no variance. Retuning NaN.")
        return np.nan

    # Find where q is close to zero. Limit set at |q| < 0.1
    # See https://github.com/LRydin/MFDFA/issues/33
    is0 = np.abs(q) < 0.1

    # Reshape q to perform np.float_power
    q_non0 = q[~is0].reshape(-1, 1)

    # Get the fluctuation function, which is a function of the windows and of q
    # When q = 2 (i.e., multifractal = False)
    # The formula is equivalent to np.sqrt(np.mean(var))
    # And corresponds to the Root Mean Square (RMS)
    fluctuation = np.float_power(np.mean(np.float_power(var, q_non0 / 2), axis=1), 1 / q_non0.T)

    if np.sum(is0) > 0:
        fluc0 = np.exp(0.5 * np.mean(np.log(var)))
        fluctuation = np.insert(fluctuation, np.where(is0)[0], [fluc0])

    return fluctuation


# =============================================================================
#  Utils MFDFA
# =============================================================================
def _singularity_spectrum(q, slopes):
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

    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """
    # Components of the singularity spectrum
    # ---------------------------------------
    # The generalised Hurst exponents `h(q)` from MFDFA, which are simply the slopes of each DFA
    # for various `q` values
    out = {"h": slopes}

    # The generalised Hurst exponent h(q) is related to the Scaling Exponent Tau t(q):
    out["Tau"] = q * slopes - 1

    # Calculate Singularity Exponent H or α, which needs tau
    out["H"] = np.gradient(out["Tau"]) / np.gradient(q)

    # Calculate Singularity Dimension Dq or f(α), which needs tau and q
    # The relation between α and f(α) (H and D) is called the Multifractal (MF) spectrum or
    # singularity spectrum, which resembles the shape of an inverted parabola.
    out["D"] = q * out["H"] - out["Tau"]

    # Features (Orozco-Duque et al., 2015)
    # ---------------------------------------
    # The width of the MSP quantifies the degree of the multifractality
    # the spectrum width delta quantifies the degree of the multifractality.
    out["Width"] = np.nanmax(out["H"]) - np.nanmin(out["H"])

    # the singularity exponent, for which the spectrum D takes its maximum value (α0)
    out["Peak"] = out["H"][np.nanargmax(out["D"])]

    # The mean of the maximum and minimum values of singularity exponent H
    out["Mean"] = (np.nanmax(out["H"]) + np.nanmin(out["H"])) / 2

    # The value of singularity spectrum D, corresponding to the maximum value of
    # singularity exponent H, indicates the maximum fluctuation of the signal.
    out["Max"] = out["D"][np.nanargmax(out["H"])]

    # The vertical distance between the singularity spectrum *D* where the singularity
    # exponents are at their minimum and maximum.
    out["Delta"] = out["D"][np.nanargmax(out["H"])] - out["D"][np.nanargmin(out["H"])]

    # the asymmetric ratio (AR) defined as the ratio between h calculated with negative q and the
    # total width of the spectrum. If the multifractal spectrum is symmetric, AR should be equal to
    # 0.5
    out["Asymmetry"] = (np.nanmin(out["H"]) - out["Peak"]) / out["Width"]

    # h-fluctuation index (hFI), which is defined as the power of the second derivative of h(q)
    # hFI tends to zero in high fractionation signals.
    if len(slopes) > 3:
        # Help needed to double check that!
        out["Fluctuation"] = np.sum(np.gradient(np.gradient(out["h"])) ** 2) / (
            2 * np.max(np.abs(q)) + 2
        )
    else:
        out["Fluctuation"] = np.nan
    # hFI tends to zero in high fractionation signals. hFI has no reference point when a set of
    # signals is evaluated, so hFI must be normalisedd, so that hFIn = 1 is the most organised and
    # the most regular signal in the set
    # BUT the formula in Orozco-Duque (2015) is weird, as HFI is a single value so cannot be
    # normalized by its range...

    # Faini (2021): new index that describes the distribution of the generalized Hurst's exponents
    # without requiring the Legendre transform. This index, αCF, is the cumulative function of the
    # squared increments of the generalized Hurst's exponents between consecutive moment orders.
    out["Increment"] = np.sum(np.gradient(slopes) ** 2 / np.gradient(q))

    return out


# =============================================================================
#  Plots
# =============================================================================
def _fractal_dfa_plot(info, scale, fluctuations):

    polyfit = np.polyfit(np.log2(scale), np.log2(fluctuations), 1)
    fluctfit = 2 ** np.polyval(polyfit, np.log2(scale))
    plt.loglog(scale, fluctuations, "o", c="#90A4AE")
    plt.xlabel(r"$\log_{2}$(Scale)")
    plt.ylabel(r"$\log_{2}$(Fluctuations)")
    plt.loglog(scale, fluctfit, c="#E91E63", label=r"$\alpha$ = {:.3f}".format(info["Alpha"]))

    plt.legend(loc="lower right")
    plt.title("Detrended Fluctuation Analysis (DFA)")

    return None


def _fractal_mdfa_plot(info, scale, fluctuations, q):

    # Prepare figure
    fig = plt.figure(constrained_layout=False)
    spec = matplotlib.gridspec.GridSpec(ncols=2, nrows=2)

    ax_fluctuation = fig.add_subplot(spec[0, 0])
    ax_spectrum = fig.add_subplot(spec[0, 1])
    ax_tau = fig.add_subplot(spec[1, 0])
    ax_hq = fig.add_subplot(spec[1, 1])

    n = len(q)
    colors = plt.cm.viridis(np.linspace(0, 1, n))

    for i in range(n):
        # Plot the points
        ax_fluctuation.loglog(
            scale,
            fluctuations[:, i],
            "o",
            fillstyle="full",
            markeredgewidth=0.0,
            c=colors[i],
            alpha=0.2,
            markersize=6,
            base=2,
            zorder=1,
        )

        # Plot the polyfit line
        polyfit = np.polyfit(np.log2(scale), np.log2(fluctuations[:, i]), 1)
        fluctfit = 2 ** np.polyval(polyfit, np.log2(scale))
        ax_fluctuation.loglog(scale, fluctfit, c=colors[i], base=2, label="_no_legend_", zorder=2)

        # Add labels for max and min
        if i == 0:
            ax_fluctuation.plot(
                [],
                label=f"$h$ = {polyfit[0]:.3f}, $q$ = {q[0]:.1f}",
                c=colors[0],
            )
        elif i == (n - 1):
            ax_fluctuation.plot(
                [],
                label=f"$h$ = {polyfit[0]:.3f}, $q$ = {q[-1]:.1f}",
                c=colors[-1],
            )

    ax_fluctuation.set_xlabel(r"$\log_{2}$(Scale)")
    ax_fluctuation.set_ylabel(r"$\log_{2}$(Fluctuations)")
    ax_fluctuation.legend(loc="lower right")

    # Plot the singularity spectrum
    # ax.set_title("Singularity Spectrum")
    ax_spectrum.set_ylabel(r"Singularity Dimension ($D$)")
    ax_spectrum.set_xlabel(r"Singularity Exponent ($H$)")
    ax_spectrum.axvline(
        x=info["Peak"],
        color="black",
        linestyle="dashed",
        label=r"Peak = {:.3f}".format(info["Peak"]),
    )
    ax_spectrum.plot(
        [np.nanmin(info["H"]), np.nanmax(info["H"])],
        [np.nanmin(info["D"])] * 2,
        color="red",
        linestyle="solid",
        label=r"Width = {:.3f}".format(info["Width"]),
    )
    ax_spectrum.plot(
        [np.nanmin(info["H"]), np.nanmax(info["H"])],
        [info["D"][-1], info["D"][0]],
        color="#B71C1C",
        linestyle="dotted",
        label=r"Delta = {:.3f}".format(info["Delta"]),
    )
    ax_spectrum.plot(info["H"], info["D"], "o-", c="#FFC107")
    ax_spectrum.legend(loc="lower right")

    # Plot tau against q
    # ax.set_title("Scaling Exponents")
    ax_tau.set_ylabel(r"Scaling Exponent ($τ$)")
    ax_tau.set_xlabel(r"Fractal Exponent ($q$)")
    ax_tau.plot(q, info["Tau"], "o-", c="#E91E63")

    # Plot H against q
    # ax.set_title("Generalised Hurst Exponents")
    ax_hq.set_ylabel(r"Singularity Exponent ($H$)")
    ax_hq.set_xlabel(r"Fractal Exponent ($q$)")
    ax_hq.plot(q, info["H"], "o-", c="#2196F3")

    fig.suptitle("Multifractal Detrended Fluctuation Analysis (MFDFA)")

    return None
