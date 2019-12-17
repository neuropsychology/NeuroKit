# -*- coding: utf-8 -*-
import numpy as np


def _embed(signal, order=3, delay=1):
    """Time-delay embedding.

    Adapted from `EntroPy <https://github.com/raphaelvallat/entropy>`_, check it out!

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    order : int
        Embedding dimension (order).
    delay : int
        Delay.

    Returns
    -------
    array
        Embedded time-series, of shape (n_times - (order - 1) * delay, order)

    Examples
    ---------
    >>> signal = np.cos(np.linspace(start=0, stop=20, num=1000))
    >>> _embed(signal)
    """
    N = len(signal)

    # Sanity checks
    if order * delay > N:
        raise ValueError("NeuroKit error: _embedding(): order * delay should be lower than length of signal.")
    if delay < 1:
        raise ValueError("NeuroKit error: _embedding(): 'delay' has to be at least 1.")
    if order < 2:
        raise ValueError("NeuroKit error: _embedding(): 'order' has to be at least 2.")

    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = signal[i * delay:i * delay + Y.shape[1]]

    return(Y.T)
