# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np




def entropy_shannon(signal):
    """Compute the Shannon entrop (SE).

    Entropy is a measure of unpredictability of the state, or equivalently, of its average information content. Shannon entropy (SE) is one of the first and most basic measure of entropy and a foundational concept of information theory. Shannon’s entropy quantifies the amount of information in a variable.


    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.


    Returns
    ----------
    float
        The Shannon entropy as float value.

    See Also
    --------
    entropy_approximate, entropy_sample, entropy_fuzzy

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=30, num=100))
    >>> nk.entropy_shannon(signal)
    6.6438561897747395


    References
    -----------
    - `pyEntropy` <https://github.com/nikdon/pyEntropy>`_
    - `EntroPy` <https://github.com/raphaelvallat/entropy>`_
    - `nolds` <https://github.com/CSchoel/nolds>`_
    """
    # Check if string
    if not isinstance(signal, str):
        signal = list(signal)

    signal = np.array(signal)

    # Create a frequency data
    data_set = list(set(signal))
    freq_list = []
    for entry in data_set:
        counter = 0.
        for i in signal:
            if i == entry:
                counter += 1
        freq_list.append(float(counter) / len(signal))

    # Shannon entropy
    shannon_entropy = 0.0
    for freq in freq_list:
        shannon_entropy += freq * np.log2(freq)
    shannon_entropy = -shannon_entropy

    return shannon_entropy
