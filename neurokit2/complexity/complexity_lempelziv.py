# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..signal.signal_binarize import _signal_binarize_threshold
from .complexity_embedding import complexity_embedding
from .utils import _get_coarsegrained, _get_scale


def complexity_lempelziv(
    signal,
    delay=1,
    dimension=2,
    scale="default",
    method="median",
    permutation=False,
    multiscale=False,
    normalize=True,
    show=False,
):
    """Lempel-Ziv Complexity (LZC, PLZC and MPLZC)

    Computes Lempel-Ziv Complexity (LZC) to quantify the regularity of the signal, by scanning
    symbolic sequences for new patterns, increasing the complexity count every time a new sequence
    is detected. Regular signals have a lower number of distinct patterns and thus have low LZC
    whereas irregular signals are characterized by a high LZC. While often being interpreted as a
    complexity measure, LZC was originally proposed to reflect randomness (Lempel and Ziv, 1976).

    Permutation Lempel-Ziv Complexity (PLZC) combines permutation and LZC.
    A finite sequence of symbols is first generated (numbers of types of symbols = `dimension!`)
    and LZC is computed over the symbol series.

    Multiscale Permutation Lempel-Ziv Complexity (MPLZC) combines permutation LZC and multiscale approach.
    It first performs a coarse-graining procedure to the original time series by constructing the coarse-grained
    time series in non-overlapping windows of increasing length (scale) where the number of data points are
    averaged. PLZC is then computed for each scaled series.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common
        to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or
        to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
        Only relevant if `permutation = True`. If `multiscale = True`, a delay of 1 (see Borowska, 2021)
        is used for coarsegraining.
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns
        an array with two columns corresponding to the original signal and its delayed (by Tau) version.
        Only relevant if `permutation = True` or `multiscale = True`.
    scale : str or int or list
        A list of scale factors used for coarse graining the time series. If 'default', will use
        ``range(len(signal) / (dimension + 10))`` (see discussion
        `here <https://github.com/neuropsychology/NeuroKit/issues/75#issuecomment-583884426>`_).
        If 'max', will use all scales until half the length of the signal. If an integer, will create
        a range until the specified int. Only relevant if `multiscale = True`.
    method : str
        Method for partitioning the signal into a binary sequence.
        Current options are "median" (default) or "mean", where each data point is assigned 0
        if lower than the median or mean of signal respectively, and 1 if higher.
        Only relevant if `permutation = False`.
    permutation : bool
        If True, returns Permutation Lempel-Ziv Complexity (PLZC; Bai et al., 2015).
    multiscale : bool
        Returns the multiscale permutation LZC (MPLZC; Borowska et al., 2021).
    normalize : bool
        Defaults to True, to obtain a complexity measure independent of sequence length.
    show : bool
        Show the MPLZC values for each scale factor (only if `multiscale = True`).

    Returns
    ----------
    lzc : float
        Lempel Ziv Complexity (LZC). Returns the mean value of PLZC over scale factors
        if multiscale permutation LZC (MPLZC) is computed.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute LZC.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)
    >>>
    >>> # LZC
    >>> lzc, info = nk.complexity_lempelziv(signal, method="median")
    >>> lzc #doctest: +SKIP
    >>>
    >>> # PLZC
    >>> plzc, info = nk.complexity_lempelziv(signal, delay=1, dimension=3, permutation=True)
    >>> plzc #doctest: +SKIP
    >>>
    >>> # MPLZC
    >>> mplzc, info = nk.complexity_lempelziv(signal, delay=1, dimension=3, multiscale=True, show=True)
    >>> mplzc #doctest: +SKIP

    References
    ----------
    - Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences. IEEE Transactions on
    information theory, 22(1), 75-81.
    - Nagarajan, R. (2002). Quantifying physiological data with Lempel-Ziv complexity-certain
    issues. IEEE Transactions on Biomedical Engineering, 49(11), 1371–1373. doi:10.1109/tbme.2002.804582.
    - Kaspar, F., & Schuster, H. G. (1987). Easily calculable measure for the complexity of
    spatiotemporal patterns. Physical Review A, 36(2), 842.
    - Zhang, Y., Hao, J., Zhou, C., & Chang, K. (2009). Normalized Lempel-Ziv complexity and
    its application in bio-sequence analysis. Journal of mathematical chemistry, 46(4), 1203-1212.
    - Bai, Y., Liang, Z., & Li, X. (2015). A permutation Lempel-Ziv complexity measure for EEG
    analysis. Biomedical Signal Processing and Control, 19, 102-114.
    - Borowska, M. (2021). Multiscale Permutation Lempel–Ziv Complexity Measure for Biomedical
    Signal Analysis: Interpretation and Application to Focal EEG Signals. Entropy, 23(7), 832.

    """

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Prepare info dict
    if multiscale:
        info = {"Normalize": normalize, "type": "MPLZC"}
    elif permutation:
        info = {"Normalize": normalize, "type": "PLZC"}
    else:
        info = {"Normalize": normalize, "type": "LZC"}

    # Run
    lzc, info = _complexity_lempelziv(
        signal,
        delay=delay,
        dimension=dimension,
        method=method,
        normalize=normalize,
        permutation=permutation,
        multiscale=multiscale,
        scale_factors=scale,
        show=show,
    )
    info.update(info)

    return lzc, info


# =============================================================================
# Utilities
# =============================================================================


def _complexity_lempelziv(
    signal,
    delay=1,
    dimension=2,
    method="median",
    normalize=True,
    permutation=False,
    multiscale=False,
    scale_factors="default",
    show=False,
):

    if multiscale:
        # MPLZC
        # apply coarsegraining procedure
        scale_factors = _get_scale(signal, scale="default", dimension=dimension)
        # Permutation for each scaled series
        lzc = np.zeros(len(scale_factors))
        for i, tau in enumerate(scale_factors):
            y = _get_coarsegrained(signal, scale=tau, force=False)
            sequence = _complexity_lempelziv_permutation(y, delay=1, dimension=dimension)
            lzc[i] = _complexity_lempelziv_count(
                sequence, normalize=normalize, permutation=True, dimension=dimension
            )
        info = {
            "Dimension": dimension,
            "Delay": delay,
            "Scale": scale_factors,
            "Values": lzc,
            "SD": np.std(lzc),
        }
        complexity = np.mean(lzc)
        if show:
            _complexity_lempelziv_multiscale_plot(scale_factors, lzc)

    elif permutation:
        # PLZC
        sequence = _complexity_lempelziv_permutation(signal, delay=delay, dimension=dimension)
        complexity = _complexity_lempelziv_count(
            sequence, normalize=normalize, permutation=True, dimension=dimension
        )
        info = {"Dimension": dimension, "Delay": delay}

    else:
        # for normal LZC
        sequence = _signal_binarize_threshold(np.asarray(signal), threshold=method).astype(int)
        complexity = _complexity_lempelziv_count(sequence, normalize=normalize, permutation=False)
        info = {"Method": method}

    return complexity, info


def _complexity_lempelziv_multiscale_plot(scale_factors, lzc_values):

    fig = plt.figure(constrained_layout=False)
    fig.suptitle("Permutation LZC (PLZC) values across scale factors")
    plt.ylabel("PLZC values")
    plt.xlabel("Scale")
    plt.plot(scale_factors, lzc_values, color="#FF9800")

    return fig


def _complexity_lempelziv_permutation(signal, delay=1, dimension=2):
    """Permutation on the signal (i.e., converting to ordinal pattern)."""
    # Time-delay embedding
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    # Sort the order of permutations
    embedded = embedded.argsort(kind="quicksort")
    sequence = np.unique(embedded, return_inverse=True, axis=0)[1]

    return sequence


def _complexity_lempelziv_count(sequence, normalize=True, permutation=False, dimension=2):
    """Computes LZC counts from symbolic sequences"""
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
                if w >= n:
                    break
                else:
                    u = 0
                    v = 1
                    v_max = 1
            else:
                v = 1

    if normalize is True:
        if permutation is False:
            out = (complexity * np.log2(n)) / n
        else:
            out = (complexity * np.log(n) / np.log(np.math.factorial(dimension))) / n

    return out
