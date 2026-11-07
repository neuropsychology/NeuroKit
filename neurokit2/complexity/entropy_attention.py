from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal

from ..misc import NeuroKitWarning


def entropy_attention(signal, show=False, silent=False, base=2, **kwargs):
    """**Attention Entropy (AttEn)**

    Yang et al. (2020) propose a conceptually new approach called **Attention Entropy (AttEn)**,
    which pays attention only to the key observations (local maxima and minima; i.e., peaks).
    Instead of counting the frequency of all observations, it analyzes the frequency distribution
    of the intervals between the key observations in a time-series. The advantages of the attention
    entropy are that it does not need any parameter to tune, is robust to the time-series length,
    and requires only linear time to compute.

    Because this index relies on peak-detection, it is not suited for noisy signals. Signal
    cleaning (in particular filtering), and eventually more tuning for the peak detection
    algorithm, can help.

    **AttEn** is computed as the average of the :func:`Shannon entropies <entropy_shannon>` of the
    distributions of four types of intervals (in samples):

    * **MaxMax**: The entropy of local-maxima intervals.
    * **MinMin**: The entropy of local-minima intervals.
    * **MaxMin**: The entropy of intervals between local maxima and subsequent minima.
    * **MinMax**: The entropy of intervals between local minima and subsequent maxima.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    show : bool
        If True, the local maxima and minima will be displayed.
    silent : bool
        If ``True``, silence possible warnings.
    base : float
        The logarithmic base to use, defaults to ``2``, giving a unit in *bits* (as in
        :func:`entropy_shannon`). Use ``np.e`` for *nats*.
    **kwargs
        Other arguments to be passed to ``scipy.signal.find_peaks()``. They are used for both the
        maxima (on the signal) and the minima (on the inverted signal), so that, for instance,
        ``height=0.5`` keeps the minima below -0.5.

    Returns
    --------
    atten : float
        The attention entropy of the signal. It is NaN if one type of interval is missing (e.g.,
        with fewer than two local maxima).
    info : dict
        A dictionary containing values of sub-entropies, such as ``MaxMax``, ``MinMin``,
        ``MaxMin``, and ``MinMax``.

    See Also
    --------
    entropy_shannon, entropy_cumulative_residual

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=1, frequency=5, noise=0.1)

      # Compute Attention Entropy
      @savefig p_entropy_attention1.png scale=100%
      atten, info = nk.entropy_attention(signal, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      atten


    References
    -----------
    * Yang, J., Choudhary, G. I., Rahardja, S., & Franti, P. (2020). Classification of interbeat
      interval time-series using attention entropy. IEEE Transactions on Affective Computing.

    """
    # Note: Code is based on the EntropyHub's package (whose peak detection, PkFind, gives the same
    # key patterns as scipy.signal.find_peaks())

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    signal = np.asarray(signal, dtype=float)

    # Identify key patterns
    Xmax, _ = scipy.signal.find_peaks(signal, **kwargs)
    Xmin, _ = scipy.signal.find_peaks(-signal, **kwargs)

    # Intervals from each key pattern to the next one of each type. The maxima and minima alternate
    # with the default peak detection, but not necessarily when it is tuned (e.g., with `distance`)
    intervals = {
        "AttEn_MaxMax": np.diff(Xmax),
        "AttEn_MinMin": np.diff(Xmin),
        "AttEn_MaxMin": _entropy_attention_next(Xmax, Xmin),
        "AttEn_MinMax": _entropy_attention_next(Xmin, Xmax),
    }
    info = {key: _entropy_attention_shannon(t, base) for key, t in intervals.items()}
    atten = np.mean(list(info.values()))

    if np.isnan(atten) and silent is False:
        warn(
            "Not enough local maxima and minima were detected to compute all the intervals of AttEn. Returning np.nan",
            category=NeuroKitWarning,
        )

    if show is True:
        plt.plot(signal, zorder=0, c="black")
        plt.scatter(Xmax, signal[Xmax], c="green", zorder=1)
        plt.scatter(Xmin, signal[Xmin], c="red", zorder=2)

    return atten, info


def _entropy_attention_next(a, b):
    """Intervals from each key pattern in a to the next key pattern in b (both sorted)."""
    idx = np.searchsorted(b, a, side="right")
    valid = idx < len(b)
    return b[idx[valid]] - a[valid]


def _entropy_attention_shannon(intervals, base=2):
    """Shannon entropy of the distribution of the intervals (NaN if there are none)."""
    if len(intervals) == 0:
        return np.nan
    p = np.bincount(intervals)
    p = p[p > 0] / len(intervals)
    return -np.sum(p * np.log(p)) / np.log(base)
