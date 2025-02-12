from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal

from ..misc import NeuroKitWarning


def entropy_attention(signal, show=False, silent=False, **kwargs):
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

    **AttEn** is computed as the average of various subindices, such as:

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
    **kwargs
        Other arguments to be passed to ``scipy.signal.find_peaks()``.

    Returns
    --------
    atten : float
        The attention entropy of the signal.
    info : dict
        A dictionary containing values of sub-entropies, such as ``MaxMax``, ``MinMin``,
        ``MaxMin``, and ``MinMax``.
    **kwargs
        Other arguments to be passed to ``scipy.signal.find_peaks()``.

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
    # Note: Code is based on the EntropyHub's package

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Identify key patterns
    Xmax, _ = scipy.signal.find_peaks(signal, **kwargs)
    Xmin, _ = scipy.signal.find_peaks(-signal, **kwargs)

    if len(Xmax) == 0 or len(Xmin) == 0:
        if silent is False:
            warn(
                "No local maxima or minima was detected, which makes it impossible to compute AttEn"
                ". Returning np.nan",
                category=NeuroKitWarning,
            )
        return np.nan, {}

    Txx = np.diff(Xmax)
    Tnn = np.diff(Xmin)
    Temp = np.diff(np.sort(np.hstack((Xmax, Xmin))))

    if Xmax[0] < Xmin[0]:
        Txn = Temp[::2]
        Tnx = Temp[1::2]
    else:
        Txn = Temp[1::2]
        Tnx = Temp[::2]

    Edges = np.arange(-0.5, len(signal) + 1)
    Pnx, _ = np.histogram(Tnx, Edges)
    Pnn, _ = np.histogram(Tnn, Edges)
    Pxx, _ = np.histogram(Txx, Edges)
    Pxn, _ = np.histogram(Txn, Edges)

    Pnx = Pnx[Pnx != 0] / len(Tnx)
    Pxn = Pxn[Pxn != 0] / len(Txn)
    Pnn = Pnn[Pnn != 0] / len(Tnn)
    Pxx = Pxx[Pxx != 0] / len(Txx)

    maxmax = -sum(Pxx * np.log(Pxx))
    maxmin = -sum(Pxn * np.log(Pxn))
    minmax = -sum(Pnx * np.log(Pnx))
    minmin = -sum(Pnn * np.log(Pnn))
    Av4 = np.mean([minmin, maxmax, maxmin, minmax])

    if show is True:
        plt.plot(signal, zorder=0, c="black")
        plt.scatter(Xmax, signal[Xmax], c="green", zorder=1)
        plt.scatter(Xmin, signal[Xmin], c="red", zorder=2)

    return Av4, {
        "AttEn_MaxMax": maxmax,
        "AttEn_MinMin": minmin,
        "AttEn_MaxMin": maxmin,
        "AttEn_MinMax": minmax,
    }


# def _find_keypatterns(signal):
#     """This original function seems to be equivalent to scipy.signal.find_peaks()"""
#     n = len(signal)
#     vals = np.zeros(n)
#     for i in range(1, n - 1):
#         if signal[i - 1] < signal[i] > signal[i + 1]:
#             vals[i] = i

#         elif signal[i - 1] < signal[i] == signal[i + 1]:
#             k = 1
#             while (i + k) < n - 1 and signal[i] == signal[i + k]:
#                 k += 1

#             if signal[i] > signal[i + k]:
#                 vals[i] = i + ((k - 1) // 2)

#     return vals[vals != 0].astype(int)
