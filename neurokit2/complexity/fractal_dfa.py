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

    For stationary signals, :math:`\\alpha` estimates the Hurst exponent, a measure of long-range
    autocorrelation: :math:`\\alpha = 0.5` for uncorrelated fluctuations (white noise),
    :math:`0.5 < \\alpha < 1` for long-range correlations (persistence), :math:`\\alpha < 0.5` for
    anti-correlations (anti-persistence), and :math:`\\alpha = 1` for 1/f (pink) noise. Values
    above 1 indicate non-stationary signals: :math:`\\alpha = 1.5` for a Brownian motion (a random
    walk), and :math:`\\alpha = H + 1` for a fractional Brownian motion of Hurst exponent *H*.

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
      spectrum: the distance from the peak to the singularity exponent *H* of the most negative
      *q*, as a proportion of the distance between the *H* of the most negative and most positive
      *q*. AR = 0.5 indicates that the multifractal spectrum is symmetric (Orozco-Duque et al.,
      2015).
    * **Fluctuation**: The *h*-fluctuation index (hFI) is defined as the power of the second
      derivative of *h*\\(q): the sum of the squared second differences of *h*\\(q) (for unit
      steps of *q*), divided by :math:`2|q_{max}| + 2` (Orozco-Duque et al., 2015). It tends to
      zero for highly fractionated signals. Orozco-Duque et al. (2015) then normalise it across a
      set of signals with their min-max range.
    * **Increment**: The cumulative function of the squared increments (:math:`\\alpha CF`) of the
      generalized Hurst's exponents between consecutive moment orders is a more robust index of the
      distribution of the generalized Hurst's exponents (Faini et al., 2021). Each squared
      increment is divided by the step between the two *q* values, which gives Faini's index for
      unit steps.

    This function can be called either via ``fractal_dfa()`` or ``complexity_dfa()``, and its
    multifractal variant can be directly accessed via ``fractal_mfdfa()`` or ``complexity_mfdfa()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    scale : list
        A list containing the lengths of the windows (number of data points in each subseries) that
        the signal is divided into. Also referred to as Tau :math:`\\tau`. If ``"default"``, will
        set it to a logarithmic scale (so that each window scale has the same weight) with a
        minimum of 10 and maximum of a tenth of the length (to have more than 10 windows to
        calculate the average fluctuation), with consecutive scales about 10% apart (about 24 per
        decade). If an integer, the number of scales between these bounds.
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
        The sequence of fractal exponents when ``multifractal=True`` (sorted in increasing order).
        For *q* = 0 (and :math:`|q| < 0.1`), the fluctuation function is the logarithmic average
        of Kantelhardt et al. (2002). Setting ``q = 2`` (default for DFA) gives a result of a
        standard DFA. For instance, Ihlen (2012) uses ``q = [-5, -3, -1, 0, 1, 3, 5]`` (default
        when for multifractal). In general, positive q moments amplify the contribution of fractal
        components with larger amplitude and negative q moments amplify the contribution of
        fractal with smaller amplitude (Kantelhardt et al., 2002). Segments whose detrended
        variance is numerically zero (at the rounding error of the signal, e.g., runs of identical
        values) are left out, as they would make the negative q moments infinite.
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
    * Peng, C. K., Buldyrev, S. V., Havlin, S., Simons, M., Stanley, H. E., & Goldberger, A. L.
      (1994). Mosaic organization of DNA nucleotides. Physical Review E, 49(2), 1685-1689.
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
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

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

    # Detrended variances at or below the rounding error of the profile are numerically zero
    # (segments that are exactly polynomial, e.g., runs of identical values). This is relative to
    # the signal's amplitude, so that the results do not depend on its units (#1208), and allows
    # for the rounding error accumulated by the cumulative sum.
    tolerance = (n * np.finfo(float).eps * np.max(np.abs(signal))) ** 2

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
        fluctuations[i] = _fractal_dfa_fluctuation(segments, trends, q, tolerance=tolerance)

    if len(fluctuations) == 0:
        return np.nan, info

    # Max. DFA ---------------------
    if maxdfa is True:
        # Find knees of fluctuations (in log2 of the scale)
        knee = np.zeros(fluctuations.shape[1])
        for i in range(fluctuations.shape[1]):
            knee[i] = find_knee(y=np.log2(fluctuations[:, i]), x=np.log2(scale), show=False, verbose=False)
        # Cut fluctuations
        keep = np.log2(scale) <= np.nanmax(knee)
        fluctuations = fluctuations[keep, :]
        scale = scale[keep]
        info["scale"] = scale
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

    # Default scale number: as many as keeps consecutive scales about 10% apart, which is as close as
    # they can be while remaining distinct (and evenly log-spaced) once rounded to whole samples
    if scale is None or isinstance(scale, str):
        scale = int(np.abs(np.log(max(int(n / 10), 1) / 10)) / np.log(1.1)) + 1

    # See https://github.com/neuropsychology/NeuroKit/issues/206
    if isinstance(scale, (int, np.integer)):
        scale = np.exp(np.linspace(np.log(10), np.log(max(int(n / 10), 1)), scale))
        scale = np.unique(np.round(scale).astype(int))  # keep only unique

    # Sanity checks (return warning for too short scale)
    if len(scale) < 2:
        raise ValueError("NeuroKit error: more than one window is needed. Increase 'scale'.")

    if np.min(scale) < 2:
        raise ValueError("NeuroKit error: there must be at least 2 data points in each window. Decrease 'scale'.")
    if np.max(scale) >= n:
        raise ValueError("NeuroKit error: the window cannot contain more data points than the time series. Decrease 'scale'.")

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

    # Fractal powers as floats, sorted (the singularity spectrum is computed as a derivative along q)
    q = np.unique(np.asarray_chkfinite(q, dtype=float))

    return q


def _slopes(scale, fluctuations, q):
    # Extract the slopes of each `q` power obtained with MFDFA to later produce
    # either the singularity spectrum or the multifractal exponents.
    # Note: added by Leonardo Rydin (see https://github.com/LRydin/MFDFA/)

    # Ensure mfdfa has the same q-power entries as q
    if fluctuations.shape[1] != q.shape[0]:
        raise ValueError("Fluctuation function and q powers don't match in dimension.")

    # Allocated array for slopes
    slopes = np.full(len(q), np.nan)
    # if fluctuations are zero (or NaN when all the segments of a scale are numerically zero),
    # these scales are left out of the fit
    with np.errstate(divide="ignore", invalid="ignore"):
        log_fluctuations = np.log2(fluctuations)
    # Find slopes of each q-power
    for i in range(len(q)):
        finite = np.isfinite(log_fluctuations[:, i])
        if np.sum(finite) >= 2:
            slopes[i] = np.polyfit(np.log2(scale)[finite], log_fluctuations[finite, i], 1)[0]

    return slopes


def _fractal_dfa_getwindow(signal, n, window, overlap=True):
    # Windows starting every half window (or every window), up to the last one that fits
    starts = np.arange(0, n - window + 1, window // 2 if overlap else window)
    return np.lib.stride_tricks.sliding_window_view(signal, window)[starts]


def _fractal_dfa_trends(segments, window, order=1):
    # Least-squares polynomial trends of all the segments at once, from their polynomial
    # coefficients on the time axis (scaled to [-1, 1] for numerical conditioning). Computing the
    # coefficients first avoids forming the (window x window) projection matrix.
    x = np.vander(np.linspace(-1, 1, window), order + 1)
    return (segments @ np.linalg.pinv(x).T) @ x.T


def _fractal_dfa_fluctuation(segments, trends, q=2, tolerance=0):
    # Detrend
    detrended = segments - trends

    # Compute variance
    var = np.var(detrended, axis=1)

    # Remove where var is numerically zero (at or below the rounding error of the signal). Any
    # cutoff above that (e.g., relative to the largest variance) would drop genuinely small
    # fluctuations, which dominate the negative q moments of multifractal signals.
    var = var[var > tolerance]
    if len(var) == 0:
        warn("All detrended segments have no variance. Retuning NaN.")
        return np.nan

    # Find where q is close to zero. Limit set at |q| < 0.1
    # See https://github.com/LRydin/MFDFA/issues/33
    is0 = np.abs(q) < 0.1

    # Get the fluctuation function, which is a function of the windows and of q
    # When q = 2 (i.e., multifractal = False)
    # The formula is equivalent to np.sqrt(np.mean(var))
    # And corresponds to the Root Mean Square (RMS)
    q_non0 = np.where(is0, 1, q).reshape(-1, 1)
    fluctuation = np.float_power(np.mean(np.float_power(var, q_non0 / 2), axis=1), 1 / q_non0[:, 0])

    # For q = 0, the logarithmic average (Kantelhardt et al., 2002)
    fluctuation[is0] = np.exp(0.5 * np.mean(np.log(var)))

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
    needed (by default `q = [-5, -3, -1, 0, 1, 3, 5]`).

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

    # Calculate Singularity Exponent H or α, which needs tau (its derivative along q, also when the
    # q values are unevenly spaced)
    out["H"] = np.gradient(out["Tau"], q)

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
    # total width of the spectrum (Orozco-Duque et al., 2015, eq. 16): (h_r - h_max) / (h_r - h_i),
    # with h_r and h_i the singularity exponents at the most negative and most positive q. If the
    # multifractal spectrum is symmetric, AR should be equal to 0.5
    out["Asymmetry"] = (out["H"][0] - out["Peak"]) / (out["H"][0] - out["H"][-1])

    # h-fluctuation index (hFI), which is defined as the power of the second derivative of h(q)
    # (Orozco-Duque et al., 2015, eq. 17): the sum of the squared second differences of h(q) over
    # unit steps of q, divided by 2 * |q_max| + 2. For other steps, the second derivative is
    # estimated with divided differences, each weighted by its step.
    if len(slopes) > 3:
        step = (q[2:] - q[:-2]) / 2
        d2h = np.diff(np.diff(slopes) / np.diff(q)) / step
        out["Fluctuation"] = np.sum(d2h**2 * step) / (2 * np.max(np.abs(q)) + 2)
    else:
        out["Fluctuation"] = np.nan
    # hFI tends to zero in high fractionation signals. As it has no reference point, Orozco-Duque et
    # al. (2015) normalise it across a set of signals, (hFI - min(hFI)) / (max(hFI) - min(hFI)), so
    # that 1 is the most organised and regular signal in the set.

    # Faini (2021): new index that describes the distribution of the generalized Hurst's exponents
    # without requiring the Legendre transform. This index, αCF, is the cumulative function of the
    # squared increments of the generalized Hurst's exponents between consecutive moment orders:
    # sum of (h(q) - h(q - 1))^2 for unit steps of q. Dividing by the step keeps it comparable for
    # other steps (it approximates the integral of h'(q)^2).
    out["Increment"] = np.sum(np.diff(slopes) ** 2 / np.diff(q))

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
