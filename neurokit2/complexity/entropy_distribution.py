import numpy as np
import pandas as pd
import scipy.stats

from .utils_complexity_embedding import complexity_embedding
from .entropy_shannon import entropy_shannon


def entropy_distribution(signal=None, delay=1, dimension=3, bins="Sturges", base=2):
    """**Distribution Entropy (DistrEn)**

    Distribution Entropy (**DistrEn**, more commonly known as **DistEn**).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    bins : int or str
        Method to find the number of bins. Can be a number, or one of ``"Sturges"``, ``"Rice"``,
        ``"Doane"``, or ``"sqrt"``.
    base : int
        The logarithmic base to use for :func:`entropy_shannon`.

    Returns
    --------
    distren : float
        The Distance Entropy entropy of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_shannon

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=5)

      distren, info = nk.entropy_distribution(signal)
      distren

    References
    -----------
    * Li, P., Liu, C., Li, K., Zheng, D., Liu, C., & Hou, Y. (2015). Assessing the complexity of
      short-term heartbeat interval series by distribution entropy. Medical & biological
      engineering & computing, 53(1), 77-87.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Store parameters
    info = {
        "Dimension": dimension,
        "Delay": delay,
        "Bins": bins,
    }

    # Time-delay embedding
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)

    # Compute distance
    n = len(embedded)
    d = np.zeros(round(n * (n - 1) / 2))
    for k in range(1, n):
        Ix = (int((k - 1) * (n - k / 2)), int(k * (n - ((k + 1) / 2))))
        d[Ix[0] : Ix[1]] = np.max(
            abs(np.tile(embedded[k - 1, :], (n - k, 1)) - embedded[k:, :]), axis=1
        )
    # TODO: "D is symmetrical. Only the upper or lower triangular matrix will actually be adequate
    # for the estimation of the ePDF, which can be used to facilitate its fast calculation."

    n_d = len(d)

    # Number of bins
    if isinstance(bins, str):
        bins = bins.lower()
        if bins == "sturges":
            n_bins = np.ceil(np.log2(n_d) + 1)
        elif bins == "rice":
            n_bins = np.ceil(2 * (n_d ** (1 / 3)))
        elif bins == "sqrt":
            n_bins = np.ceil(np.sqrt(n_d))
        elif bins == "doanes":
            sigma = np.sqrt(6 * (n_d - 2) / ((n_d + 1) * (n_d + 3)))
            n_bins = np.ceil(1 + np.log2(n_d) + np.log2(1 + abs(scipy.stats.skew(d) / sigma)))
        else:
            raise Exception("Please enter a valid binning method")
    else:
        n_bins = bins

    # Get probability
    freq, _ = np.histogram(d, int(n_bins))
    freq = freq / freq.sum()

    # Compute Shannon Entropy
    distren, _ = entropy_shannon(freq=freq, base=base)

    # Normalize by number of bins (so that the range should be within [0, 1])
    distren = distren / (np.log(n_bins) / np.log(base))

    return distren, info
