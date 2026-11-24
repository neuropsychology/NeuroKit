from warnings import warn

import numpy as np
import pandas as pd
import scipy.special

from ..misc import NeuroKitWarning
from .entropy_shannon import entropy_shannon


def entropy_ofentropy(signal, scale=10, bins=10, base=2):
    """**Entropy of entropy (EnofEn)**

    Entropy of entropy (EnofEn or EoE) combines the features of :func:`MSE <entropy_multiscale>`
    with an alternate measure of information, called *superinformation*, used in DNA sequencing
    (Hsu et al., 2017).

    1. The signal is divided into consecutive non-overlapping windows of ``scale`` (:math:`\\tau`)
       samples (the last samples that do not fill a window are left out).
    2. The range of the signal (from its minimum to its maximum) is divided into ``bins``
       (:math:`s_1`) slices of equal width, and the :func:`Shannon entropy <entropy_shannon>` of
       the distribution of the samples of each window among these slices is computed. Its average
       over the windows is returned as ``AvEn``.
    3. These entropies can only take a finite number of values (levels), which depends on
       ``scale`` and ``bins``. EnofEn is the Shannon entropy of the distribution of the entropies of
       the windows among these levels, i.e., it measures how much the information content of the
       signal changes over time.

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
    base : float
        The logarithmic base of both Shannon entropies, defaults to ``2``, giving a unit in *bits*
        (use ``np.e`` for *nats*). It does not change the levels of the window entropies, so
        EnofEn and ``AvEn`` are only multiplied by a constant.

    Returns
    --------
    enofen : float
        The Entropy of entropy of the signal. It is NaN (with a warning) for signals with missing
        values or shorter than ``scale``.
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
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    if scale < 1 or bins < 1:
        raise ValueError(
            f"NeuroKit error: entropy_ofentropy(): `scale` and `bins` should be at least 1 (got {scale} and {bins})."
        )

    info = {"Scale": scale, "Bins": bins}
    scale, bins = int(scale), int(bins)
    signal = np.asarray(signal, dtype=float)

    # divide a one-dimensional discrete time series of length n into consecutive
    # non-overlapping windows w where each window is of length 'scale'
    n_windows = len(signal) // scale

    if n_windows == 0 or np.any(~np.isfinite(signal)):
        warn(
            "The signal contains missing values or is shorter than `scale`. EnofEn cannot be computed and is returned as NaN.",
            category=NeuroKitWarning,
        )
        info["AvEn"] = np.nan
        return np.nan, info

    # Divide the range (of the whole signal) into s1 slices into n equal width bins corresponding
    # to a discrete state k
    sigrange = (np.min(signal), np.max(signal))
    edges = np.linspace(sigrange[0], sigrange[1], bins + 1)
    signal = signal[: n_windows * scale]

    # Compute the probability for a sample in each window to occur in state k (the same bins as
    # np.histogram(): left-closed, except the last one which includes the maximum)
    states = np.minimum(np.searchsorted(edges, signal, side="right") - 1, bins - 1)
    windows = np.repeat(np.arange(n_windows), scale)
    freq = np.bincount(windows * bins + states, minlength=n_windows * bins).reshape(n_windows, bins)

    # Next, we calculate the Shannon entropy value of each window.
    shanens = np.sum(scipy.special.entr(freq / scale), axis=1) / np.log(base)
    info["AvEn"] = np.mean(shanens)

    # Number of unique ShanEn values (depending on the scale)
    _, freq2 = np.unique(np.round(shanens, 12), return_counts=True)
    freq2 = freq2 / freq2.sum()

    # Shannon entropy again to measure the degree of the "changing"
    enofen, _ = entropy_shannon(freq=freq2, base=base)

    return enofen, info
