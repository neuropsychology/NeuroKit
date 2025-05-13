# -*- coding: utf-8 -*-
import math

import numpy as np
import pandas as pd

from .utils_complexity_ordinalpatterns import complexity_ordinalpatterns
from .utils_complexity_symbolize import complexity_symbolize


def complexity_lempelziv(
    signal,
    delay=1,
    dimension=2,
    permutation=False,
    symbolize="mean",
    **kwargs,
):
    """**Lempel-Ziv Complexity (LZC, PLZC and MSLZC)**

    Computes Lempel-Ziv Complexity (LZC) to quantify the regularity of the signal, by scanning
    symbolic sequences for new patterns, increasing the complexity count every time a new sequence
    is detected. Regular signals have a lower number of distinct patterns and thus have low LZC
    whereas irregular signals are characterized by a high LZC. While often being interpreted as a
    complexity measure, LZC was originally proposed to reflect randomness (Lempel and Ziv, 1976).

    Permutation Lempel-Ziv Complexity (**PLZC**) combines LZC with :func:`permutation <entropy_permutation>`.
    A sequence of symbols is generated from the permutations observed in the :func:`tine-delay
    embedding <complexity_embedding>`, and LZC is computed over it.

    Multiscale (Permutation) Lempel-Ziv Complexity (**MSLZC** or **MSPLZC**) combines permutation
    LZC with the :func:`multiscale approach <entropy_multiscale>`. It first performs a
    :func:`coarse-graining <complexity_coarsegraining>` procedure to the original time series.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter. Only used
        when ``permutation=True``.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter. Only used
        when ``permutation=True``
    permutation : bool
        If ``True``, will return PLZC.
    symbolize : str
        Only used when ``permutation=False``. Method to convert a continuous signal input into a
        symbolic (discrete) signal. By default, assigns 0 and 1 to values below and above the mean.
        Can be ``None`` to skip the process (in case the input is already discrete). See
        :func:`complexity_symbolize` for details.
    **kwargs
        Other arguments to be passed to :func:`complexity_ordinalpatterns` (if
        ``permutation=True``) or :func:`complexity_symbolize`.

    Returns
    ----------
    lzc : float
        Lempel Ziv Complexity (LZC) of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute LZC.

    See Also
    --------
    .complexity_symbolize, .complexity_ordinalpatterns, .entropy_permutation,

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)

      # LZC
      lzc, info = nk.complexity_lempelziv(signal)
      lzc

      # PLZC
      plzc, info = nk.complexity_lempelziv(signal, delay=1, dimension=3, permutation=True)
      plzc

    .. ipython:: python

      # MSLZC
      @savefig p_complexity_lempelziv1.png scale=100%
      mslzc, info = nk.entropy_multiscale(signal, method="LZC", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      # MSPLZC
      @savefig p_complexity_lempelziv2.png scale=100%
      msplzc, info = nk.entropy_multiscale(signal, method="LZC", permutation=True, show=True)
      @suppress
      plt.close()

    References
    ----------
    * Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences. IEEE Transactions on
      information theory, 22(1), 75-81.
    * Nagarajan, R. (2002). Quantifying physiological data with Lempel-Ziv complexity-certain
      issues. IEEE Transactions on Biomedical Engineering, 49(11), 1371-1373.
    * Kaspar, F., & Schuster, H. G. (1987). Easily calculable measure for the complexity of
      spatiotemporal patterns. Physical Review A, 36(2), 842.
    * Zhang, Y., Hao, J., Zhou, C., & Chang, K. (2009). Normalized Lempel-Ziv complexity and
      its application in bio-sequence analysis. Journal of mathematical chemistry, 46(4), 1203-1212.
    * Bai, Y., Liang, Z., & Li, X. (2015). A permutation Lempel-Ziv complexity measure for EEG
      analysis. Biomedical Signal Processing and Control, 19, 102-114.
    * Borowska, M. (2021). Multiscale Permutation Lempel-Ziv Complexity Measure for Biomedical
      Signal Analysis: Interpretation and Application to Focal EEG Signals. Entropy, 23(7), 832.

    """

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    # Store parameters
    info = {"Permutation": permutation}

    # Permutation or not
    if permutation:
        info["Dimension"] = dimension
        info["Delay"] = delay
        # Permutation on the signal (i.e., converting to ordinal pattern).
        _, info = complexity_ordinalpatterns(signal, delay=delay, dimension=dimension, **kwargs)
        symbolic = info["Uniques"]
    else:
        # Binarize the signal
        symbolic = complexity_symbolize(signal, method=symbolize, **kwargs)

    # Count using the lempelziv algorithm
    info["Complexity_Kolmogorov"], n = _complexity_lempelziv_count(symbolic)

    # Normalize
    if permutation is False:
        lzc = (info["Complexity_Kolmogorov"] * np.log2(n)) / n
    else:
        lzc = (info["Complexity_Kolmogorov"] * np.log2(n) / np.log2(math.factorial(dimension))) / n

    return lzc, info


# =============================================================================
# Utilities
# =============================================================================
def _complexity_lempelziv_count(symbolic):
    """Computes LZC counts from symbolic sequences."""

    # TODO: I really can't imagine that there is no faster way of doing that that with a while loop

    # Convert to string (faster)
    string = "".join(list(symbolic.astype(int).astype(str)))

    # Initialize variables
    n = len(string)

    s = "0" + string
    c = 1
    j = 1
    i = 0
    k = 1
    k_max = 1
    stop = False

    # Start counting
    while stop is False:
        if s[i + k] != s[j + k]:
            if k > k_max:
                # k_max stores the length of the longest pattern in the LA that has been matched
                # somewhere in the SB
                k_max = k

            # we increase i while the bit doesn't match, looking for a previous occurrence of a
            # pattern. s[i+k] is scanning the "search buffer" (SB)
            i = i + 1
            # we stop looking when i catches up with the first bit of the "look-ahead" (LA) part.
            if i == j:
                # If we were actually compressing, we would add the new token here. here we just
                # count reconstruction STEPs
                c = c + 1
                # we move the beginning of the LA to the end of the newly matched pattern.
                j = j + k_max
                # if the LA surpasses length of string, then we stop.
                if j + 1 > n:
                    stop = True
                # after STEP,
                else:
                    # we reset the searching index to beginning of SB (beginning of string)
                    i = 0
                    # we reset pattern matching index. Note that we are actually matching against
                    # the first bit of the string, because we added an extra 0 above, so i+k is the
                    # first bit of the string.
                    k = 1
                    # and we reset max length of matched pattern to k.
                    k_max = 1
            else:
                # we've finished matching a pattern in the SB, and we reset the matched pattern
                # length counter.
                k = 1
        # I increase k as long as the pattern matches, i.e. as long as s[j+k] bit string can be
        # reconstructed by s[i+k] bit string. Note that the matched pattern can "run over" j
        # because the pattern starts copying itself (see LZ 76 paper). This is just what happens
        # when you apply the cloning tool on photoshop to a region where you've already cloned...
        else:
            k = k + 1
            # if we reach the end of the string while matching, we need to add that to the tokens,
            # and stop.
            if j + k > n:
                c = c + 1
                stop = True

    return c, n
