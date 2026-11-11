from warnings import warn

import numpy as np
import pandas as pd
import scipy.spatial
import scipy.stats

from ..misc import NeuroKitWarning
from .entropy_shannon import entropy_shannon
from .utils_complexity_embedding import complexity_embedding


def entropy_distribution(signal=None, delay=1, dimension=3, bins="Sturges", base=2):
    """**Distribution Entropy (DistrEn)**

    Distribution Entropy (**DistrEn**, more commonly known as **DistEn**) quantifies the complexity
    of a signal from the distribution of the distances between all its embedded vectors (Li et al.,
    2015). Unlike :func:`SampEn <entropy_sample>`, it does not rely on a tolerance to count the
    similar vectors, and it was proposed to be more stable for short signals.

    1. The signal is :func:`time-delay embedded <complexity_embedding>`.
    2. The Chebyshev distances (the largest absolute difference between their elements) between all
       the pairs of vectors are computed, leaving out the distance of each vector to itself. As the
       distance matrix is symmetric, each pair is counted once.
    3. The empirical probability distribution of the distances, :math:`p_t`, is estimated with a
       histogram of *M* bins (``bins``) spanning the range of the distances.
    4. DistrEn is the :func:`Shannon entropy <entropy_shannon>` of this distribution, divided by its
       maximum :math:`\\log(M)` so that it lies between 0 and 1.

    .. math::

        DistrEn = -\\frac{1}{\\log_2(M)} \\sum_{t=1}^{M} p_t \\log_2(p_t)

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
        The number of bins *M* of the histogram (at least 2), or the method to find it from the
        number of distances: ``"Sturges"`` (default), ``"Rice"``, ``"Doane"`` (or ``"Doanes"``),
        or ``"sqrt"``.
    base : int
        The logarithmic base to use for :func:`entropy_shannon` (which does not change the
        normalized DistrEn).

    Returns
    --------
    distren : float
        The Distribution Entropy of the signal, between 0 and 1. It is NaN (with a warning) for
        signals with missing values, or too short to give at least 2 distances.
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
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    methods = ["sturges", "rice", "doane", "doanes", "sqrt"]
    if isinstance(bins, str):
        if bins.lower() not in methods:
            raise ValueError(
                f"NeuroKit error: entropy_distribution(): bins method '{bins}' not recognized. It should be"
                + " a number or one of 'Sturges', 'Rice', 'Doane' or 'sqrt'."
            )
    elif bins < 2:
        raise ValueError(f"NeuroKit error: entropy_distribution(): `bins` should be at least 2, not {bins}.")

    # Store parameters
    info = {
        "Dimension": dimension,
        "Delay": delay,
        "Bins": bins,
    }

    # Time-delay embedding
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)

    # Chebyshev distances between all the pairs of vectors (i < j). The distance matrix D is
    # symmetrical, so only its upper triangular part (without the diagonal of self-distances) is
    # needed for the estimation of the ePDF. (The missing values are checked beforehand, as
    # pdist() leaves them out of the Chebyshev distance.)
    d = scipy.spatial.distance.pdist(embedded, metric="chebyshev")
    n_d = len(d)

    if n_d < 2 or np.any(~np.isfinite(embedded)):
        warn(
            "The signal contains missing values or is too short (fewer than 2 distances between its"
            " embedded vectors). DistrEn cannot be computed and is returned as NaN.",
            category=NeuroKitWarning,
        )
        return np.nan, info

    # All the distances are the same (e.g., constant signal): they fall into one bin
    if np.all(d == d[0]):
        return 0.0, info

    # Number of bins
    if isinstance(bins, str):
        bins = bins.lower()
        if bins == "sturges":
            n_bins = np.ceil(np.log2(n_d) + 1)
        elif bins == "rice":
            n_bins = np.ceil(2 * (n_d ** (1 / 3)))
        elif bins == "sqrt":
            n_bins = np.ceil(np.sqrt(n_d))
        else:  # Doane
            sigma = np.sqrt(6 * (n_d - 2) / ((n_d + 1) * (n_d + 3)))
            # (with 2 distances, sigma is 0 and so is the skewness)
            skewness = abs(scipy.stats.skew(d) / sigma) if n_d > 2 else 0
            n_bins = np.ceil(1 + np.log2(n_d) + np.log2(1 + skewness))
    else:
        n_bins = bins
    n_bins = int(n_bins)

    # Get probability
    freq, _ = np.histogram(d, n_bins)
    freq = freq / freq.sum()

    # Compute Shannon Entropy
    distren, _ = entropy_shannon(freq=freq, base=base)

    # Normalize by number of bins (so that the range should be within [0, 1])
    distren = distren / (np.log(n_bins) / np.log(base))

    return distren, info
