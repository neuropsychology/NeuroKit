import numpy as np
import scipy.signal
import scipy.stats
from matplotlib import pyplot as plt


def signal_autocor(
    signal, lag=None, demean=True, method="auto", unbiased=False, show=False
):
    """**Autocorrelation (ACF)**

    Compute the autocorrelation of a signal.

    Parameters
    -----------
    signal : Union[list, np.array, pd.Series]
        Vector of values.
    lag : int
        Time lag. If specified, one value of autocorrelation between signal with its lag self will
        be returned.
    demean : bool
        If ``True``, the mean of the signal will be subtracted from the signal before ACF
        computation.
    method : str
        Using ``"auto"`` runs ``scipy.signal.correlate`` to determine the faster algorithm. Other
        methods are kept for legacy reasons, but are not recommended. Other methods include
        ``"correlation"`` (using :func:`.np.correlate`) or ``"fft"`` (Fast Fourier Transform).
    show : bool
        If ``True``, plot the autocorrelation at all values of lag.

    Returns
    -------
    r : float
        The cross-correlation of the signal with itself at different time lags. Minimum time lag is
        0, maximum time lag is the length of the signal. Or a correlation value at a specific lag
        if lag is not ``None``.
    info : dict
        A dictionary containing additional information, such as the confidence interval.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Example 1: Using 'Correlation' Method
      signal = [1, 2, 3, 4, 5]
      @savefig p_signal_autocor1.png scale=100%
      r, info = nk.signal_autocor(signal, show=True, method='correlation')
      @suppress
      plt.close()

    .. ipython:: python

      # Example 2: Using 'FFT' Method
      signal = nk.signal_simulate(duration=5, sampling_rate=100, frequency=[5, 6], noise=0.5)
      @savefig p_signal_autocor2.png scale=100%
      r, info = nk.signal_autocor(signal, lag=2, method='fft', show=True)
      @suppress
      plt.close()

    .. ipython:: python

      # Example 3: Using 'unbiased' Method
      @savefig p_signal_autocor3.png scale=100%
      r, info = nk.signal_autocor(signal, lag=2, method='fft', unbiased=True, show=True)
      @suppress
      plt.close()

    """
    n = len(signal)

    # Demean
    if demean:
        signal = np.asarray(signal) - np.nanmean(signal)

    # Run autocor
    method = method.lower()
    if method == "unbiased":
        unbiased = True
        method = "fft"

    if method in ["auto"]:
        acov = scipy.signal.correlate(signal, signal, mode="full", method="auto")[
            n - 1 :
        ]
    elif method in ["cor", "correlation", "correlate"]:
        acov = np.correlate(signal, signal, mode="full")
        acov = acov[n - 1 :]  # Min time lag is 0
    elif method == "fft":
        a = np.concatenate((signal, np.zeros(n - 1)))  # added zeros to your signal
        fft = np.fft.fft(a)
        acf = np.fft.ifft(np.conjugate(fft) * fft)[:n]
        acov = acf.real
    else:
        raise ValueError("Method must be 'auto', 'correlation' or 'fft'.")

    # If unbiased, normalize by the number of overlapping elements
    if unbiased is True:
        normalization = np.arange(n, 0, -1)
        acov /= normalization

    # Normalize (so that max correlation is 1)
    r = acov / np.max(acov)

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
        plt.ylim(-1.1, 1.1)

    if lag is not None:
        if lag > n:
            raise ValueError(
                "NeuroKit error: signal_autocor(): The time lag exceeds the duration of the signal. "
            )
        else:
            r = r[lag]

    return r, {"CI_low": ci_low, "CI_high": ci_high, "Method": method, "ACov": acov}
