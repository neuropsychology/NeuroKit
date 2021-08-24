# -*- coding: utf-8 -*-
import numpy as np


def complexity_lempelziv(signal, threshold="median", normalize=True):
    """
    Computes Lempel Ziv Complexity (LZC) to quantify the regularity of the signal, by scanning
    symbolic sequences for new patterns, increasing the complexity count every time a new sequence is detected.
    Regular signals have a lower number of distinct patterns and thus have low LZC whereas irregular signals are
    characterized by a high lZC.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    threshold : str
        Method for partitioning the signal into a binary sequence.
        Current options are "median" (default) or "mean", where each data point is assigned 0
        if lower than the median or mean of signal respecitvely, and 1 if higher.
    normalize : bool
        Defaults to True, to obtain a complexity measure independent of sequence length.

    Returns
    ----------
    float
        Lempel Ziv Complexity.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5, noise=10)
    >>>
    >>> lzc = nk.complexity_lempelziv(signal, threshold="median")
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
    """

    # convert signal into binary sequence
    p_seq = _complexity_lempelziv_binarize(signal, threshold=threshold)

    # pre-set variables
    complexity = 1
    n = len(p_seq)
    pointer = 0
    current_prefix_len = 1
    current_substring_len = 1
    final_substring_len = 1

    # iterate over sequence
    while current_prefix_len + current_substring_len <= n:
        if (p_seq[pointer + current_substring_len - 1] == p_seq[current_prefix_len + current_substring_len - 1]):
            current_substring_len += 1
        else:
            final_substring_len = max(current_substring_len, final_substring_len)
            pointer += 1
            if pointer == current_prefix_len:
                complexity += 1
                current_prefix_len = current_prefix_len + final_substring_len
                current_substring_len = 1
                pointer = 0
                final_substring_len = 1
            else:
                current_substring_len = 1

    if current_substring_len != 1:
        complexity += 1

    if normalize is True:
        complexity = _complexity_lempelziv_normalize(p_seq, complexity)

    return complexity


def _complexity_lempelziv_binarize(signal, threshold="median"):

    # method to convert signal by
    if threshold == "median":
        threshold = np.median(signal)
    elif threshold == "mean":
        threshold = np.mean(signal)

    p_seq = signal.copy()
    # convert
    for index, value in enumerate(signal):
        if value < threshold:
            p_seq[index] = 0
        else:
            p_seq[index] = 1
    p_seq = p_seq.astype(int)

    return p_seq

def _complexity_lempelziv_normalize(sequence, complexity):

    n = len(sequence)
    upper_bound = n / np.log2(n)

    return complexity / upper_bound
