# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def entropy_shannon(signal):
    """Shannon entropy (SE)

    Python implementation of Shannon entropy (SE). Entropy is a measure of unpredictability of the state,
    or equivalently, of its average information content. Shannon entropy (SE) is one of the first and
    most basic measure of entropy and a foundational concept of information theory. Shannon’s entropy
    quantifies the amount of information in a variable.

    This function can be called either via ``entropy_shannon()`` or ``complexity_se()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.


    Returns
    ----------
    shanen : float
        The Shannon entropy of the single time series, or the mean ShEn
        across the channels of an n-dimensional time series.
    parameters : dict
        A dictionary containing additional information regarding the parameters used
        to compute Shannon entropy and the individual ShEn values of each
        channel if an n-dimensional time series is passed.

    See Also
    --------
    entropy_approximate, entropy_sample, entropy_fuzzy

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> entropy, parameters = nk.entropy_shannon(signal)
    >>> entropy #doctest: +SKIP


    References
    -----------
    - `pyEntropy` <https://github.com/nikdon/pyEntropy>`_

    - `EntroPy` <https://github.com/raphaelvallat/entropy>`_

    - `nolds` <https://github.com/CSchoel/nolds>`_

    """
    # prepare parameters
    parameters = {}

    # sanitize input
    if signal.ndim > 1:
        # n-dimensional
        if not isinstance(signal, (pd.DataFrame, np.ndarray)):
            raise ValueError(
            "NeuroKit error: entropy_shannon(): your n-dimensional data has to be in the",
            " form of a pandas DataFrame or a numpy ndarray.")
        if isinstance(signal, np.ndarray):
            # signal.shape has to be in (len(channels), len(samples)) format
            signal = pd.DataFrame(signal).transpose()

        shen_values = []
        for i, colname in enumerate(signal):
            channel = np.array(signal[colname])
            shen = _entropy_shannon(channel)
            shen_values.append(shen)
        parameters['values'] = shen_values
        out = np.mean(shen_values)

    else:
        # if one signal time series        
        out = _entropy_shannon(signal)

    return out, parameters


def _entropy_shannon(signal):

    # Check if string
    if not isinstance(signal, str):
        signal = list(signal)

    signal = np.array(signal)

    # Create a frequency data
    data_set = list(set(signal))
    freq_list = []
    for entry in data_set:
        counter = 0.0
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
