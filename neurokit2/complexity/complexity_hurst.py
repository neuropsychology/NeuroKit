import numpy as np
import pandas as pd
import scipy.special

from .fractal_dfa import _fractal_dfa_findwindows


def complexity_hurst(signal, windows="default", corrected=False, q=2, show=False):
    """Hurst Exponent (H)

    This function estimates the Hurst exponent via the standard rescaled range (R/S) approach, but other methods exist,
    such as Detrended Fluctuation Analysis (DFA, see ``fractal_dfa()``).

    The Hurst exponent is a measure for the "long-term memory" of a signal. It can be used to determine whether the time
    series is more, less, or equally likely to increase if it has increased in previous steps. This property makes the
    Hurst exponent especially interesting for the analysis of stock data. It typically ranges from 0 to 1, with 0.5
    corresponding to a Brownian motion. If H < 0.5, the time-series covers less "distance" than a random walk (the memory
    of the signal decays faster than at random), and vice versa.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
        or dataframe.
    windows : list
        A list containing the lengths of the windows (number of data points in
        each subseries). See ``fractal_dfa()`` for more information.
    corrected : boolean
        if True, the Anis-Lloyd-Peters correction factor will be applied to the
        output according to the expected value for the individual (R/S) values.

    See Also
    --------
    fractal_dfa

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>>
    >>> h, info = nk.complexity_hurst(signal, corrected=True)
    >>> h
    >>> h, info = nk.complexity_hurst(signal, show=True)

    References
    ----------
    - Brandi, G., & Di Matteo, T. (2021). On the statistics of scaling exponents and the Multiscaling Value at Risk. The European Journal of Finance, 1-22.
    - https://github.com/CSchoel/nolds

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    n = len(signal)
    windows = _fractal_dfa_findwindows(n, windows)

    # get individual values for (R/S)_n
    rs_vals = np.array([_complexity_hurst_rs(signal, window) for window in windows])

    # filter NaNs (zeros should not be possible, because if R is 0 then S is also zero)
    valid = ~np.isnan(rs_vals)
    rs_vals = rs_vals[valid]
    n_vals = np.asarray(windows)[valid]

    # it may happen that no rs vals are left (if all values of data are the same)
    if len(rs_vals) == 0:
        raise ValueError(
            "All (R/S) values are NaN. Check your data, or try a different window length."
        )

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

    return h, {"Values": n_vals, "Scores": rs_vals, "Corrected": corrected, "Intercept": poly[1]}


# =============================================================================
# Utilities
# =============================================================================


def expected_rs(n):
    """
    Calculates the expected (R/S)_n for white noise for a given n.
    This is used as a correction factor in the function hurst_rs. It uses the
    formula of Anis-Lloyd-Peters (see [h_3]_).
    """
    front = (n - 0.5) / n
    i = np.arange(1, n)
    back = np.sum(np.sqrt((n - i) / i))
    if n <= 340:
        middle = scipy.special.gamma((n - 1) * 0.5) / np.sqrt(np.pi) / scipy.special.gamma(n * 0.5)
    else:
        middle = 1.0 / np.sqrt(n * np.pi * 0.5)
    return front * middle * back


def _complexity_hurst_rs(signal, window):
    """
    Calculates an individual R/S value in the rescaled range approach for
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
    # find standard deviation
    # we should use the unbiased estimator, since we do not know the true mean
    s = np.std(seqs, ddof=1, axis=1)
    # some ranges may be zero and have to be excluded from the analysis
    idx = np.where(r != 0)
    r = r[idx]
    s = s[idx]
    # it may happen that all ranges are zero (if all values in data are equal)
    if len(r) == 0:
        return np.nan
    # return mean of r/s along subsequence index
    return np.mean(r / s)


def _complexity_hurst_generalized(signal, q=2):
    """From https://github.com/PTRRupprecht/GenHurst"""

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
            SSxx = np.sum(X ** 2) - N * mx ** 2
            my = np.sum(Y) / N
            SSxy = np.sum(np.multiply(X, Y)) - N * mx * my
            cc1 = SSxy / SSxx
            cc2 = my - cc1 * mx
            ddVd = dV - cc1
            VVVd = VV - np.multiply(cc1, np.arange(1, N + 1, dtype=np.float64)) - cc2
            mcord[tt - 1] = np.mean(np.abs(ddVd) ** q) / np.mean(np.abs(VVVd) ** q)

        mx = np.mean(np.log10(x))
        SSxx = np.sum(np.log10(x) ** 2) - Tmax * mx ** 2
        my = np.mean(np.log10(mcord))
        SSxy = np.sum(np.multiply(np.log10(x), np.transpose(np.log10(mcord)))) - Tmax * mx * my
        H[k] = SSxy / SSxx
        k = k + 1

    mH = np.mean(H) / q

    return mH
