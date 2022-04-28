import numpy as np
import pandas as pd
import scipy.cluster.vq
import scipy.special

from ..stats import standardize
from .utils_complexity_embedding import complexity_embedding


def entropy_symbolicdynamic(signal, dimension=3, c=6, method="MEP", **kwargs):
    """**Symbolic Dynamic Entropy (SyDyEn) and its Multiscale variants (MSSyDyEn)**

    Symbolic Dynamic Entropy (SyDyEn) combines the merits of symbolic dynamic and information
    theory.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension()` to estimate the optimal value for this parameter.
    c : int
        Number of symbols *c*.
    method : str
        Method for transforming the signal into a symbolic sequence. Can be one of ``"MEP"``
        (default), ``"NCDF"``, ``"linear"``, ``"uniform"``, ``"kmeans"``, ``"equal"`` or
        ``"finesort"``.
    **kwargs : optional
        Other keyword arguments (currently not used).

    Returns
    -------
    SyDyEn : float
        Symbolic Dynamic Entropy (SyDyEn) of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_shannon, entropy_multiscale, entropy_dispersion

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal
      signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)

      # Compute Symbolic Dynamic Entropy
      sydyen, info = nk.entropy_symbolicdynamic(signal, c=3, method="MEP")
      sydyen

      sydyen, info = nk.entropy_symbolicdynamic(signal, c=3, method="kmeans")
      sydyen

      # Compute Multiscale Symbolic Dynamic Entropy (MSSyDyEn)
      @savefig p_entropy_symbolicdynamic1.png scale=100%
      mssydyen, info = nk.entropy_multiscale(signal, method="MSSyDyEn", show=True)
      @suppress
      plt.close()

      # Compute Modified Multiscale Symbolic Dynamic Entropy (MMSyDyEn)
      @savefig p_entropy_symbolicdynamic2.png scale=100%
      mmsydyen, info = nk.entropy_multiscale(signal, method="MMSyDyEn", show=True)
      @suppress
      plt.close()

    References
    ----------
    * Li, Y., Yang, Y., Li, G., Xu, M., & Huang, W. (2017). A fault diagnosis scheme for planetary
      gearboxes using modified multi-scale symbolic dynamic entropy and mRMR feature selection.
      Mechanical Systems and Signal Processing, 91, 295-312.
    * Rajagopalan, V., & Ray, A. (2006). Symbolic time series analysis via wavelet-based
      partitioning. Signal processing, 86(11), 3309-3320.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Store parameters
    info = {"Dimension": dimension, "c": c, "Method": method}

    # We could technically expose the Delay, but the paper is about consecutive differences so...
    if "delay" in kwargs.keys():
        delay = kwargs["delay"]
        kwargs.pop("delay")
    else:
        delay = 1

    n = len(signal)

    # There are four main steps of SDE algorithm
    # 1. Convert the time series into the symbol time series (called symbolization).
    # 2. Construct the embedding vectors based on the symbol time series and compute the potential
    # state patterns probability
    embedded = _complexity_symbolization(
        signal,
        dimension=dimension,
        delay=delay,
        c=c,
        method=method,
        **kwargs,
    )

    # 3. Construct the state transitions and compute the probability of state transitions.
    unique = np.unique(embedded, axis=0)
    counter1 = np.zeros(len(unique))
    counter2 = np.zeros((len(unique), c))
    Bins = np.arange(0.5, c + 1.5, 1)
    for i in range(len(unique)):
        Ordx = np.any(embedded - unique[i, :], axis=1) == 0
        counter1[i] = sum(Ordx) / (n - ((dimension - 1) * delay))
        Temp = embedded[
            np.hstack((np.zeros(dimension * delay, dtype=bool), Ordx[: -(dimension * delay)])), 0
        ]
        counter2[i, :], _ = np.histogram(Temp, Bins)

    Temp = np.sum(counter2, axis=1)
    counter2[Temp > 0, :] = counter2[Temp > 0, :] / np.tile(Temp[Temp > 0], (c, 1)).transpose()
    counter2[np.isnan(counter2)] = 0

    # 4. Based on the Shannon entropy [39], we define the SDE as the sum of the state entropy and
    # the state transition entropy
    with np.errstate(divide="ignore"):
        P1 = -sum(counter1 * np.log(counter1))
        P2 = np.log(np.tile(counter1, (c, 1)).transpose() * counter2)
    P2[~np.isfinite(P2)] = 0
    sydyen = P1 - sum(counter1 * np.sum(P2, axis=1))

    # Normalize
    sydyen = sydyen / np.log(c ** (dimension + 1))

    return sydyen, info


def _complexity_symbolization(signal, delay=1, dimension=3, c=3, method="MEP", rho=1, **kwargs):
    """Transform signal into symbolic sequence"""
    n = len(signal)
    method = method.lower()
    if method == "mep":
        # Maximum Entropy Partitioning (MEP)
        Temp = np.hstack((0, np.ceil(np.arange(1, c) * len(signal) / c) - 1)).astype(int)
        symbols = np.digitize(signal, np.sort(signal)[Temp])
    elif method == "ncdf":
        symbols = np.digitize(scipy.special.ndtr(standardize(signal)), np.arange(0, 1, 1 / c))
    elif method == "linear":
        symbols = np.digitize(signal, np.arange(np.min(signal), np.max(signal), np.ptp(signal) / c))
    elif method == "uniform":
        symbols = np.zeros(len(signal))
        symbols[np.argsort(signal)] = np.digitize(np.arange(n), np.arange(0, 2 * n, n / c))
    elif method == "kmeans":
        centroids, labels = scipy.cluster.vq.kmeans2(signal, c)
        labels += 1
        xx = np.argsort(centroids) + 1
        symbols = np.zeros(n)
        for k in range(1, c + 1):
            symbols[labels == xx[k - 1]] = k
    elif method == "equal":
        ix = np.argsort(signal)
        xx = np.round(np.arange(0, 2 * n, n / c)).astype(int)
        symbols = np.zeros(n)
        for k in range(c):
            symbols[ix[xx[k] : xx[k + 1]]] = k + 1
    elif method == "finesort":
        Zx = scipy.special.ndtr((signal - np.mean(signal)) / np.std(signal))
        symbols = np.digitize(Zx, np.arange(0, 1, 1 / c))
        Ym = np.zeros((n - (dimension - 1) * delay, dimension))
        for k in range(dimension):
            Ym[:, k] = Zx[k * delay : n - ((dimension - k - 1) * delay)]
        Yi = np.floor(np.max(abs(np.diff(Ym)), axis=1) / (rho * np.std(abs(np.diff(signal)))))
    else:
        raise ValueError(
            'Method must be one of "MEP", "NCDF", "linear", "uniform", "kmeans",'
            '"equal" or "finesort".'
        )

    embedded = complexity_embedding(symbols, dimension=dimension, delay=delay)

    if method == "finesort":
        Yi = np.expand_dims(Yi, axis=1)
        embedded = np.hstack((embedded, Yi))

    return embedded
