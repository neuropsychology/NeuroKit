# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal.signal_binarize import _signal_binarize_threshold
from .complexity_embedding import complexity_embedding


def complexity_lempelziv(signal, method="median", permutation=False, delay=1, dimension=2, normalize=True):
    """
    Computes Lempel-Ziv Complexity (LZC) to quantify the regularity of the signal, by scanning
    symbolic sequences for new patterns, increasing the complexity count every time a new sequence is detected.
    Regular signals have a lower number of distinct patterns and thus have low LZC whereas irregular
    signals are characterized by a high LZC.
    
    Permutation Lempel-Zic Complexity (PLZC) combines permutation and LZC. A finite sequence of symbols
    is first generated (numbers of types of symbols = `dimension!`) and LZC is computed over the symbol series.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        Method for partitioning the signal into a binary sequence.
        Current options are "median" (default) or "mean", where each data point is assigned 0
        if lower than the median or mean of signal respectively, and 1 if higher.
        Only relevant if `permutation = False`.
    permutation : bool
        If True, returns Permutation Lempel-Ziv Complexity (PLZC).
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common
        to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or
        to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
        Only relevant if `permutation = True`.
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns
        an array with two columns corresponding to the original signal and its delayed (by Tau) version.
        Only relevant if `permutation = True`.
    normalize : bool
        Defaults to True, to obtain a complexity measure independent of sequence length.

    Returns
    ----------
    lzc : float
        Lempel Ziv Complexity (LZC) or PLZC of the single time series.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute LZC or PLZC.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=10, frequency=5, noise=10)
    >>>
    >>> lzc, info = nk.complexity_lempelziv(signal, method="median", permutation=True)
    >>> lzc #doctest: +SKIP

    References
    ----------
    - Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences. IEEE Transactions on information theory,
    22(1), 75-81.

    - Nagarajan, R. (2002). Quantifying physiological data with Lempel-Ziv complexity-certain issues.
    IEEE Transactions on Biomedical Engineering, 49(11), 1371–1373. doi:10.1109/tbme.2002.804582.

    - Kaspar, F., & Schuster, H. G. (1987). Easily calculable measure for the complexity of spatiotemporal patterns.
    Physical Review A, 36(2), 842.

    - Zhang, Y., Hao, J., Zhou, C., & Chang, K. (2009). Normalized Lempel-Ziv complexity and
    its application in bio-sequence analysis. Journal of mathematical chemistry, 46(4), 1203-1212.

    - https://en.wikipedia.org/wiki/Lempel-Ziv_complexity

    - Bai, Y., Liang, Z., & Li, X. (2015). A permutation Lempel-Ziv complexity measure for EEG analysis.
    Biomedical Signal Processing and Control, 19, 102-114.
    """

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    parameters = {"Normalize": normalize}

    if permutation is False:
        # Convert signal into binary sequence
        sequence = _signal_binarize_threshold(np.asarray(signal), threshold=method).astype(int)
        parameters["Type"] = "LZC"
        parameters["Method"] = method
    else:
        # Convert into symbols
        sequence, info = _complexity_lempelziv_permutation(signal, delay=delay, dimension=dimension)
        parameters["Type"] = "PLZC"
        parameters.update(info)

    # Compute LZC
    n = len(sequence)
    complexity = _complexity_lempelziv(sequence)

    if normalize is True:
        if permutation is False:
            out = (complexity * np.log2(n)) / n
        else:
            out = (complexity * np.log(n) / np.log(np.math.factorial(dimension))) / n

    return out, parameters


# =============================================================================
# Utilities
# =============================================================================

def _complexity_lempelziv_permutation(signal, delay=1, dimension=3):
    
    # Time-delay embedding
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)

    # Sort the order of permutations
    embedded = embedded.argsort(kind="quicksort")
    
    symbols = np.unique(embedded, return_inverse=True, axis=0)[1]
    info = {"Dimension": dimension, "Delay": delay}

    return symbols, info

def _complexity_lempelziv(sequence):

    # Convert to string (faster)
    string = "".join(list(sequence.astype(str)))

    # Initialize variables
    n = len(sequence)
    u, v, w = 0, 1, 1
    v_max = 1
    complexity = 1

    while True:
        if string[u + v - 1] == string[w + v - 1]:
            v += 1
            if w + v >= n:
                complexity += 1
                break
        else:
            if v > v_max:
                v_max = v
            u += 1
            if u == w:
                complexity += 1
                w += v_max
                if w > n:
                    break
                else:
                    u = 0
                    v = 1
                    v_max = 1
            else:
                v = 1

    return complexity
