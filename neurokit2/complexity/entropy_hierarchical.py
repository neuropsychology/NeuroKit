import numpy as np
import pandas as pd

from .complexity_embedding import complexity_embedding


def entropy_hierarchical(signal, dimension=3, n=3, method="MEP", **kwargs):
    """**Hierarchical Entropy (HEn)**

    The hierarchical entropy (HEn) analysis proposed in this paper takes into consideration the entropy of the higher frequency components of a time series.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension()` to estimate the optimal value for this parameter.
    n : int
        Number of symbols.
    method : str
        Method for symbolic sequence partitioning. Can be one of ``"MEP"`` (default),
        ``"linear"``, ``"uniform"``, ``"kmeans"``.
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
    entropy_shannon, entropy_multiscale

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal
      signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)

      # Compute Symbolic Dynamic Entropy
      sydyen, info = nk.entropy_symbolicdynamic(signal, n=3, method="MEP")
      sydyen

      sydyen, info = nk.entropy_symbolicdynamic(signal, n=3, method="kmeans")
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

      MMSDE

    References
    ----------
    * Jiang, Y., Peng, C. K., & Xu, Y. (2011). Hierarchical entropy analysis for biological
      signals. Journal of Computational and Applied Mathematics, 236(5), 728-742.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # TODO: help us.
    return None, {}

    # # Store parameters
    # info = {"Dimension": dimension}

    # # We could technically expose the Delay, but the paper is about consecutive differences so...
    # if "delay" in kwargs.keys():
    #     delay = kwargs["delay"]
    #     kwargs.pop("delay")
    # else:
    #     delay = 1

    # method = "SD"

    # method = method.lower()
    # if method == "sd":
    #     Rnew = lambda x: np.std(x)
    # elif method == "var":
    #     Rnew = lambda x: np.var(x)
    # elif method == "mean":
    #     Rnew = lambda x: np.mean(abs(x - np.mean(x)))
    # elif method == "median":
    #     Rnew = lambda x: np.median(abs(x - np.median(x)))

    # signal = np.array([0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 4, 5, 6, 7, 9, 12])
    # XX, N = Hierarchy(signal, sx=2)
    # XX.shape
    # return sydyen, info


# def Hierarchy(signal, sx):
#     N = int(2 ** np.floor(np.log2(len(signal))))
#     if np.log2(len(signal)) % 1 != 0:
#         print(
#             "Only first %d samples were used in hierarchical decomposition. \
#             \nThe last %d samples of the data sequence were ignored."
#             % (N, len(signal) - N)
#         )
#     if N / (2 ** (sx - 1)) < 8:
#         raise Exception(
#             "Data length (%d) is too short to estimate entropy at the lowest"
#             " subtree. Consider reducing the number of scales." % N
#         )

#     U = np.zeros(((2 ** sx) - 1, N))
#     U[0, :] = signal[:N]
#     p = 1
#     for k in range(sx - 1):
#         for n in range(2 ** k):
#             Temp = U[(2 ** k) + n - 1, :]
#             # U[p,1:N//2]  = (Temp[:-2:2] + Temp[1:-1:2])/2
#             # U[p+1,1:N//2]= (Temp[:-2:2] - Temp[1:-1:2])/2
#             U[p, : N // 2] = (Temp[::2] + Temp[1::2]) / 2
#             U[p + 1, : N // 2] = (Temp[::2] - Temp[1::2]) / 2
#             p += 2
#     return U, N
