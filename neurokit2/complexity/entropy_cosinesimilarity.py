from warnings import warn

import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from .entropy_range import _entropy_range_tolerance
from .utils_complexity_embedding import complexity_embedding


def entropy_cosinesimilarity(signal=None, delay=1, dimension=3, tolerance="sd"):
    """**Cosine Similarity Entropy (CoSiEn) and its multiscale variant (MSCoSiEn)**

    Cosine Similarity Entropy (CoSiEn) is based on fundamental modifications of the SampEn and the
    MSEn approaches, which makes the CoSiEn amplitude-independent and robust to spikes and short
    length of data segments, two key problems with the standard SampEn.

    Following Chanwimalueang & Mandic (2017), the median of the signal is removed (an optional
    pre-processing step in the paper, which makes the angles robust to an offset), and the
    angular distance between two embedded vectors :math:`X_i` and :math:`X_j` is computed as

    .. math::

      AngDis_{i,j} = \\frac{1}{\\pi} \\cos^{-1} \\left( \\frac{X_i \\cdot X_j}{|X_i| |X_j|} \\right)

    which lies between 0 (same direction) and 1 (opposite directions), whatever the amplitude of
    the vectors. The global probability *B* is the proportion of the pairs of vectors whose angular
    distance is at most the tolerance *r* (the pairs involving a null vector, whose angle is
    undefined, are not similar), and CoSiEn is its binary Shannon entropy (in bits), between 0 and
    1:

    .. math::

      CoSiEn = -[B \\log_2(B) + (1 - B) \\log_2(1 - B)]

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*), at least 2. See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    tolerance : float
        Tolerance (often denoted as *r*), the angular distance (between 0 and 1) under which two
        vectors are considered similar. The angular distance being normalized, the tolerance is not
        multiplied by the standard deviation (SD) of the signal. If ``"sd"`` (default), will be set
        to 0.2 (i.e., :math:`0.2 * SD` of the standardized signal). Other methods of
        :func:`complexity_tolerance` are also applied to the standardized signal. Chanwimalueang &
        Mandic (2017) recommend values between 0.05 and 0.2 (and used 0.07 with ``dimension=2``).

    Returns
    --------
    cosien : float
        The Cosine Similarity Entropy of the signal (0 when no pair of vectors, or all of
        them, are similar).
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_multiscale

    Examples
    ----------
    1. Compute Cosine Similarity Entropy (**CoSiEn**)

    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=5)

      cosien, info = nk.entropy_cosinesimilarity(signal)
      cosien

    2. Compute Multiscale Cosine Similarity Entropy (**MSCoSiEn**)

    .. ipython:: python

      mscosien, info = nk.entropy_multiscale(signal, method="MSCoSiEn")
      mscosien

    References
    -----------
    * Chanwimalueang, T., & Mandic, D. P. (2017). Cosine similarity entropy: Self-correlation-based
      complexity analysis of dynamical systems. Entropy, 19(12), 652.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    if dimension < 2:
        raise ValueError("The dimension of CoSiEn should be at least 2 (one-dimensional vectors are always aligned).")
    signal = np.asarray(signal, dtype=float)

    # Store parameters
    info = {"Dimension": dimension, "Delay": delay}

    if np.any(np.isnan(signal)):
        warn("The signal contains missing values (NaN): CoSiEn is returned as NaN.", category=NeuroKitWarning)
        info["Tolerance"] = np.nan
        return np.nan, info

    info["Tolerance"] = _entropy_range_tolerance(signal, tolerance, delay, dimension)

    # Steps from Chanwimalueang et al. (2017)
    # 1. (Optional pre-processing) Remove the offset and generate a zero median series
    signal = signal - np.median(signal)

    # 2. Construct the embedding vectors
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    n = len(embedded)
    norm = np.linalg.norm(embedded, axis=1)

    # 3. Compute angular distance for all pairwise embedding vectors (i < j), by blocks of rows (of
    # about 65,000 pairs, which fit in the cache) to limit the memory used
    # 4. Obtain the number of similar patterns when a criterion AngDis(m)i,j <= rCSE is fulfilled
    similar = 0
    block = max(1, 2**16 // n)
    with np.errstate(divide="ignore", invalid="ignore"):
        for start in range(0, n, block):
            stop = min(start + block, n)
            # Cosine similarity of the vectors of the block with themselves and the following ones
            cos = np.inner(embedded[start:stop], embedded[start:]) / np.outer(norm[start:stop], norm[start:])
            # Clip the rounding errors outside [-1, 1] (NaN, i.e., not similar, for null vectors)
            angle = np.arccos(np.clip(cos, -1, 1)) / np.pi
            # Only the pairs i < j
            similar += np.sum(np.triu(angle <= info["Tolerance"], 1))

    # 5. Compute the local probability of occurrences of similar patterns
    # 6. Compute the global probability of occurrences of similar patterns
    Bm = similar / (n * (n - 1) / 2)

    # 7. Cosine similarity entropy (binary Shannon entropy, in bits)
    if Bm in [0, 1]:
        cosien = 0
    else:
        cosien = -(Bm * np.log2(Bm)) - ((1 - Bm) * np.log2(1 - Bm))

    return cosien, info
