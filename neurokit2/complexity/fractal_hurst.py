import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special

from .fractal_dfa import _fractal_dfa_findscales


def fractal_hurst(signal, scale="default", corrected=True, show=False):
    """**Hurst Exponent (H)**

    This function estimates the Hurst exponent via the standard rescaled range (R/S) approach, but
    other methods exist, such as Detrended Fluctuation Analysis (DFA, see :func:`fractal_dfa`).

    The Hurst exponent is a measure for the "long-term memory" of a signal. It can be used to
    determine whether the time series is more, less, or equally likely to increase if it has
    increased in previous steps. This property makes the Hurst exponent especially interesting for
    the analysis of stock data. It typically ranges from 0 to 1, with 0.5 corresponding to
    uncorrelated noise (i.e., the increments of a Brownian motion, since the R/S analysis
    integrates the signal). If H < 0.5, the time series is anti-persistent (increases tend to be
    followed by decreases), and if H > 0.5 it is persistent (long-range correlated).

    The R/S approach first splits the time series into non-overlapping subseries of length n. R
    (the range of the cumulative sum of the deviations from the mean) and S (the standard
    deviation) are then calculated for each subseries and the mean is taken over all subseries
    yielding (R/S)_n. This process is repeated for several lengths *n*. The final exponent is then
    derived from fitting a straight line to the plot of :math:`log((R/S)_n)` vs :math:`log(n)`.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    scale : list
        A list containing the lengths of the windows (number of data points in each subseries) that
        the signal is divided into. See :func:`fractal_dfa` for more information.
    corrected : boolean
        If ``True``, the Anis-Lloyd correction is applied: the expected (R/S)_n of Gaussian white
        noise (Anis & Lloyd, 1976) is subtracted from each (R/S)_n in log-space, and 0.5 is added
        to the resulting slope (Weron, 2002). This removes the small-sample bias of R/S, which
        overestimates H for short windows. Note that the expected value is exact when S is the
        population standard deviation (as used here). Peters' (1994) additional factor
        :math:`(n - 0.5) / n` (used e.g. in *nolds*) approximately corresponds to using the
        sample (*n - 1*) standard deviation instead.
    show : bool
        If ``True``, returns a plot.

    See Also
    --------
    fractal_dfa

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate Signal with duration of 2s
      signal = nk.signal_simulate(duration=2, frequency=5)

      # Compute Hurst Exponent
      h, info = nk.fractal_hurst(signal, corrected=True, show=True)
      h

    References
    ----------
    * Brandi, G., & Di Matteo, T. (2021). On the statistics of scaling exponents and the
      Multiscaling Value at Risk. The European Journal of Finance, 1-22.
    * Anis, A. A., & Lloyd, E. H. (1976). The expected value of the adjusted rescaled Hurst range
      of independent normal summands. Biometrika, 63(1), 111-116.
    * Weron, R. (2002). Estimating long-range dependence: finite sample properties and confidence
      intervals. Physica A, 312(1-2), 285-299.
    * https://github.com/CSchoel/nolds

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    signal = np.asarray(signal, dtype=float)

    n = len(signal)
    scale = _fractal_dfa_findscales(n, scale)

    # get individual values for (R/S)_n
    rs_vals = np.array([_fractal_hurst_rs(signal, window) for window in scale])

    # filter NaNs (zeros should not be possible, because if R is 0 then S is also zero)
    valid = ~np.isnan(rs_vals)
    rs_vals = rs_vals[valid]
    n_vals = np.asarray(scale)[valid]

    # it may happen that no rs vals are left (if all values of data are the same)
    if len(rs_vals) == 0:
        raise ValueError("All (R/S) values are NaN. Check your data, or try a different window length.")

    # Transform RS values
    rs_vals = np.log10(rs_vals)
    if corrected:
        rs_vals -= np.log10([expected_rs(n) for n in n_vals])
    n_vals = np.log10(n_vals)

    # fit a line to the logarithm of the obtained (R/S) values
    poly = np.polyfit(n_vals, rs_vals, 1)
    h = poly[0]  # Get the slope

    if corrected:
        h = poly[0] + 0.5

    if show:
        _fractal_hurst_plot(poly, n_vals, rs_vals, corrected=corrected, ax=None)

    return h, {"Values": n_vals, "Scores": rs_vals, "Corrected": corrected, "Intercept": poly[1]}


# =============================================================================
# Utilities
# =============================================================================


def expected_rs(n):
    """Calculates the expected (R/S)_n of Gaussian white noise for a given n (Anis & Lloyd, 1976).

    The formula is exact when S is the population standard deviation. The ratio of Gamma functions
    is computed in log-space, which avoids the overflow of Gamma for n > 340 (and hence the
    asymptotic approximation used for large n by other implementations).

    https://en.wikipedia.org/wiki/Hurst_exponent#cite_note-:2-17
    """
    i = np.arange(1, n)
    back = np.sum(np.sqrt((n - i) / i))
    middle = np.exp(scipy.special.gammaln((n - 1) * 0.5) - scipy.special.gammaln(n * 0.5)) / np.sqrt(np.pi)

    return middle * back


def _fractal_hurst_rs(signal, window):
    """Calculates an individual R/S value in the rescaled range approach for
    a given window size (the size of the subseries in which data should be split).
    """
    n = len(signal)
    m = n // window  # number of sequences
    # cut values at the end of data to make the array divisible by n
    signal = signal[: n - (n % window)]
    # split remaining data into subsequences of length n
    seqs = np.reshape(signal, (m, window))
    # calculate means of subsequences
    means = np.mean(seqs, axis=1)
    # normalize subsequences by substracting mean
    y = seqs - means.reshape((m, 1))
    # build cumulative sum of subsequences
    y = np.cumsum(y, axis=1)
    # find ranges
    r = np.max(y, axis=1) - np.min(y, axis=1)
    # find standard deviation (population formula, as in the classical R/S statistic, for which
    # the expected value of Anis & Lloyd (1976) is exact)
    s = np.std(seqs, ddof=0, axis=1)
    # some ranges may be zero and have to be excluded from the analysis
    idx = np.where(r != 0)
    r = r[idx]
    s = s[idx]
    # it may happen that all ranges are zero (if all values in data are equal)
    if len(r) == 0:
        return np.nan
    # return mean of r/s along subsequence index
    return np.mean(r / s)


def _fractal_hurst_plot(poly, n_vals, rs_vals, corrected=False, ax=None):
    if ax is None:  # ax option in case more plots need to be added later
        fig, ax = plt.subplots()
        fig.suptitle("Hurst Exponent via Rescaled Range (R/S) Analysis")
    else:
        fig = None

    ax.set_ylabel(r"$log$((R/S)_n)")
    ax.set_xlabel(r"$log$(n)")

    # Plot log((R/S)_n) vs log(n)
    ax.scatter(
        n_vals,
        rs_vals,
        marker="o",
        zorder=1,
        label="_no_legend_",
    )

    fit_values = [poly[0] * i + poly[1] for i in n_vals]
    if corrected:
        label = f"corrected h = {round(poly[0] + 0.5, 2)}"
    else:
        label = f"h = {round(poly[0], 2)}"
    ax.plot(n_vals, fit_values, color="#E91E63", zorder=2, linewidth=3, label=label)
    ax.legend(loc="lower right")

    return fig


# =============================================================================
# Generalized Hurst Exponent
# =============================================================================
def _fractal_hurst_generalized(signal, q=2):
    """TO BE DONE.

    The Generalized Hurst exponent method is assesses directly the scaling properties of the time series
    via the qth-order moments of the distribution of the increments.

    Different exponents `q` are associated with different characterizations of the multi-scaling complexity of the signal.
    In contrast to the popular R/S statistics approach, it does not deal with max and min functions, and thus less sensitive
    to outliers.

    From https://github.com/PTRRupprecht/GenHurst"""

    n = len(signal)
    H = np.zeros((len(range(5, 20)), 1))
    k = 0

    for Tmax in range(5, 20):
        x = np.arange(1, Tmax + 1, 1)
        mcord = np.zeros((Tmax, 1))

        for tt in range(1, Tmax + 1):
            dV = signal[np.arange(tt, n, tt)] - signal[np.arange(tt, n, tt) - tt]
            VV = signal[np.arange(tt, n + tt, tt) - tt]
            N = len(dV) + 1
            X = np.arange(1, N + 1, dtype=np.float64)
            Y = VV

            mx = np.sum(X) / N
            SSxx = np.sum(X**2) - N * mx**2
            my = np.sum(Y) / N
            SSxy = np.sum(np.multiply(X, Y)) - N * mx * my
            cc1 = SSxy / SSxx
            cc2 = my - cc1 * mx
            ddVd = dV - cc1
            VVVd = VV - np.multiply(cc1, np.arange(1, N + 1, dtype=np.float64)) - cc2
            mcord[tt - 1] = np.mean(np.abs(ddVd) ** q) / np.mean(np.abs(VVVd) ** q)

        mx = np.mean(np.log10(x))
        SSxx = np.sum(np.log10(x) ** 2) - Tmax * mx**2
        my = np.mean(np.log10(mcord))
        SSxy = np.sum(np.multiply(np.log10(x), np.transpose(np.log10(mcord)))) - Tmax * mx * my
        H[k] = SSxy / SSxx
        k = k + 1

    mH = np.mean(H) / q

    return mH
