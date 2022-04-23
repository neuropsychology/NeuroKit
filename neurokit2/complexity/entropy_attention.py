import numpy as np
import pandas as pd


def entropy_attention(signal):
    """**Attention Entropy (AttEn)**

    Yang et al. (2020) propose a conceptually new approach called **Attention Entropy (AttEn)**,
    which pays attention only to the key observations. Instead of counting the frequency of all
    observations, it analyzes the frequency distribution of the intervals between the key
    observations in a time-series. The advantages of the attention entropy are that it does not
    need any parameter to tune, is robust to the time-series length, and requires only linear time
    to compute.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    Returns
    --------
    atten : float
        The attention entropy of the signal.
    info : dict
        A dictionary containing values of sub-entropies that are averaged to give the general
        AttEn, such as ``MaxMax`` (entropy of local-maxima intervals), ``MinMin`` (entropy of
        local-minima intervals), ``MaxMin`` (entropy of intervals between local maxima and
        subsequent minima), and ``MinMax`` (entropy of intervals between local minima and
        subsequent maxima ).

    See Also
    --------
    entropy_shannon, entropy_cumulative_residual

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=5, noise=0.1)

      # Compute Differential Entropy
      atten, info = nk.entropy_attention(signal)
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

    signal = np.squeeze(signal)
    N = signal.shape[0]

    Xmax = _find_keypatterns(signal)
    Xmin = _find_keypatterns(-signal)
    Txx = np.diff(Xmax)
    Tnn = np.diff(Xmin)
    Temp = np.diff(np.sort(np.hstack((Xmax, Xmin))))

    assert len(Xmax) > 0, "No local maxima found!"
    assert len(Xmin) > 0, "No local minima found!"

    if Xmax[0] < Xmin[0]:
        Txn = Temp[::2]
        Tnx = Temp[1::2]
    else:
        Txn = Temp[1::2]
        Tnx = Temp[::2]

    Edges = np.arange(-0.5, N + 1)
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
    Av4 = (minmin + maxmax + maxmin + minmax) / 4

    return Av4, {
        "AttEn_MaxMax": maxmax,
        "AttEn_MinMin": minmin,
        "AttEn_MaxMin": maxmin,
        "AttEn_MinMax": minmax,
    }


def _find_keypatterns(X):
    Nx = len(X)
    Indx = np.zeros(Nx)
    for n in range(1, Nx - 1):
        if X[n - 1] < X[n] > X[n + 1]:
            Indx[n] = n

        elif X[n - 1] < X[n] == X[n + 1]:
            k = 1
            while (n + k) < Nx - 1 and X[n] == X[n + k]:
                k += 1

            if X[n] > X[n + k]:
                Indx[n] = n + ((k - 1) // 2)

    Indx = Indx[Indx != 0]
    return Indx
