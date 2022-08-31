import numpy as np
import pandas as pd

from .utils_complexity_embedding import complexity_embedding
from .utils_complexity_symbolize import complexity_symbolize


def entropy_symbolicdynamic(signal, dimension=3, symbolize="MEP", c=6, **kwargs):
    """**Symbolic Dynamic Entropy (SyDyEn) and its Multiscale variants (MSSyDyEn)**

    Symbolic Dynamic Entropy (SyDyEn) combines the merits of symbolic dynamic and information
    theory.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    symbolize : str
        Method to convert a continuous signal input into a symbolic (discrete) signal. Can be one
        of ``"MEP"`` (default), ``"NCDF"``, ``"linear"``, ``"uniform"``, ``"kmeans"``, ``"equal"``,
        or others. See :func:`complexity_symbolize` for details.
    c : int
        Number of symbols *c*.
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

      signal = [2, -7, -12, 5, -1, 9, 14]

      # Simulate a Signal
      signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)

      # Compute Symbolic Dynamic Entropy
      sydyen, info = nk.entropy_symbolicdynamic(signal, c=3, symbolize="MEP")
      sydyen

      sydyen, info = nk.entropy_symbolicdynamic(signal, c=3, symbolize="kmeans")
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
    * Matilla-García, M., Morales, I., Rodríguez, J. M., & Marín, M. R. (2021). Selection of
      embedding dimension and delay time in phase space reconstruction via symbolic dynamics.
      Entropy, 23(2), 221.
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
    info = {"Dimension": dimension, "c": c, "Symbolization": symbolize}

    # We could technically expose the Delay, but the paper is about consecutive differences so...
    if "delay" in kwargs.keys():
        delay = kwargs["delay"]
        kwargs.pop("delay")
    else:
        delay = 1

    n = len(signal)

    # There are four main steps of SDE algorithm
    # 1. Convert the time series into the symbol time series (called symbolization).
    symbolic = complexity_symbolize(signal, method=symbolize, c=c)

    # 2. Construct the embedding vectors based on the symbol time series and compute the potential
    # state patterns probability
    embedded = complexity_embedding(symbolic, dimension=dimension, delay=delay)

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
