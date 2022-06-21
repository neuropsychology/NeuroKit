import numpy as np
import pandas as pd

from .entropy_shannon import entropy_shannon


def entropy_ofentropy(signal, scale=10, bins=10, **kwargs):
    """**Entropy of entropy (EnofEn)**

    Entropy of entropy (EnofEn or EoE) combines the features of :func:`MSE <entropy_multiscale>`
    with an alternate measure of information, called *superinformation*, used in DNA sequencing.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    scale : int
        The size of the windows that the signal is divided into. Also referred to as Tau
        :math:`\\tau`, it represents the scale factor and corresponds to
        the amount of coarsegraining.
    bins : int
        The number of equal-size bins to divide the signal's range in.
    **kwargs : optional
        Other keyword arguments, such as the logarithmic ``base`` to use for
        :func:`entropy_shannon`.

    Returns
    --------
    enofen : float
        The Entropy of entropy of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used, such as the
        average entropy ``AvEn``.

    See Also
    --------
    entropy_shannon, entropy_multiscale

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal
      signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)

      # EnofEn
      enofen, _ = nk.entropy_ofentropy(signal, scale=10, bins=10)
      enofen

    References
    -----------
    * Hsu, C. F., Wei, S. Y., Huang, H. P., Hsu, L., Chi, S., & Peng, C. K. (2017). Entropy of
      entropy: Measurement of dynamical complexity for biological systems. Entropy, 19(10), 550.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    info = {"Scale": scale, "Bins": bins}

    # divide a one-dimensional discrete time series of length n into consecutive
    # non-overlapping windows w where each window is of length 'scale'
    n_windows = int(np.floor(len(signal) / scale))
    windows = np.reshape(signal[: n_windows * scale], (n_windows, scale))

    # Divide the range into s1 slices into n equal width bins corresponding to a discrete state k
    sigrange = (np.min(signal), np.max(signal))
    edges = np.linspace(sigrange[0], sigrange[1], bins + 1)

    # Compute the probability for a sample in each window to occur in state k
    freq = [np.histogram(windows[w, :], edges)[0] for w in range(n_windows)]
    # Next, we calculate the Shannon entropy value of each window.
    shanens = [entropy_shannon(freq=w / w.sum(), **kwargs)[0] for w in freq]
    info["AvEn"] = np.nanmean(shanens)

    # Number of unique ShanEn values (depending on the scale)
    _, freq2 = np.unique(np.round(shanens, 12), return_counts=True)
    freq2 = freq2 / freq2.sum()

    # Shannon entropy again to measure the degree of the "changing"
    enofen, _ = entropy_shannon(freq=freq2, **kwargs)

    return enofen, info
