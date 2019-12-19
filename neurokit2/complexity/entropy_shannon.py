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


    Example
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Download example EEG signal
    >>> signal = pd.read_csv('https://raw.github.com/neuropsychology/NeuroKit/master/data/example_eeg.txt', header=None)[0].values
    >>> nk.entropy_shannon(signal)
    7.566810239706894


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

    return(shannon_entropy)
