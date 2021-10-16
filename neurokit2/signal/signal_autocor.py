import numpy as np
import scipy.stats
from matplotlib import pyplot as plt


def signal_autocor(signal, lag=None, method="correlation", show=False):
    """Autocorrelation (ACF)

    Compute the autocorrelation of a signal.

    Parameters
    -----------
    signal : Union[list, np.array, pd.Series]
        Vector of values.
    lag : int
        Time lag. If specified, one value of autocorrelation between signal with its lag self will be returned.
    method : str
        Can be 'correlation' (using ``np.correlate``) or 'fft' (using FFT).
    show : bool
        If True, plot the autocorrelation at all values of lag.

    Returns
    -------
    r : float
        The cross-correlation of the signal with itself at different time lags. Minimum time lag is 0,
        maximum time lag is the length of the signal. Or a correlation value at a specific lag if lag
        is not None.
    info : dict
        A dictionary containing additional information, such as the confidence interval.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> signal = [1, 2, 3, 4, 5]
    >>> r, info = nk.signal_autocor(signal, show=True)
    >>> r #doctest: +SKIP
    >>>
    >>> signal = [3, 5, 1, 3, 1, 6, 3, 1]
    >>> r, info = nk.signal_autocor(signal, lag=2, method='fft', show=True)

    """
    n = len(signal)
    # Standardize
    signal = (np.asarray(signal) - np.nanmean(signal)) / np.nanstd(signal)

    if method.lower() == "correlation":
        r = np.correlate(signal, signal, mode="full")
        r = r[n - 1 :]  # Min time lag is 0
    elif method.lower() == "fft":
        a = np.concatenate((signal, np.zeros(n - 1)))  # added zeros to your signal
        A = np.fft.fft(a)
        S = np.conj(A) * A
        c_fourier = np.fft.ifft(S)
        r = c_fourier[: (c_fourier.size // 2) + 1].real

    # Normalize
    r = r / r[0]

    # Confidence interval
    varacf = 1.0 / n
    interval = scipy.stats.norm.ppf(1 - 0.05 / 2.0) * np.sqrt(varacf)
    ci_low, ci_high = r - interval, r + interval

    # Plot
    if show:
        plt.axhline(y=0, color="grey", linestyle="--")
        plt.plot(np.arange(1, len(r) + 1), r, lw=2)
        plt.ylabel("Autocorrelation r")
        plt.xlabel("Lag")
        plt.ylim(-1, 1)
        plt.show()

    if lag is not None:
        if lag > n:
            raise ValueError(
                "NeuroKit error: signal_autocor(): The time lag exceeds the duration of the signal. "
            )
        else:
            r = r[lag]

    return r, {"CI_low": ci_low, "CI_high": ci_high, "Method": method}
