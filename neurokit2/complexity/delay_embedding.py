# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


def delay_embedding(signal, delay=1, dimension=3, show=False):
    """Time-delay embedding of a time series (a signal).

    Takens's (1981) embedding theorem suggests that a sequence of measurements from a dynamic system includes in itself all the information required to completely reconstruct the state space (the set of all possible states of a dynamical system). Delay coordinate embedding attempts to identify the state s of the system at some time t by searching the past history of observations for similar states, and, by studying the evolution of similar states, infer information about the future of the system.

    This function is adapted from `EntroPy <https://github.com/raphaelvallat/entropy>`_ and is equivalent to
    the `delay_embedding()` function from 'nolds'.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    delay : int
        Time delay (tau).
    order : int
        Embedding dimension (m), sometimes referred to as 'order'.

    Returns
    -------
    array
        Embedded time-series, of shape (n_times - (order - 1) * delay, order)

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>>
    >>> embedded = nk.delay_embedding(signal, delay=1, show=True)
    >>> embedded = nk.delay_embedding(signal, delay=50, show=True)
    """
    N = len(signal)

    # Sanity checks
    if dimension * delay > N:
        raise ValueError("NeuroKit error: delay_embedding(): dimension * delay should be lower than length of signal.")
    if delay < 1:
        raise ValueError("NeuroKit error: delay_embedding(): 'delay' has to be at least 1.")
    if dimension < 2:
        raise ValueError("NeuroKit error: delay_embedding(): 'dimension' has to be at least 2.")

    Y = np.zeros((dimension, N - (dimension - 1) * delay))
    for i in range(dimension):
        Y[i] = signal[i * delay:i * delay + Y.shape[1]]
    embedded = Y.T

    if show is True:
        _delay_embedding_plot(embedded)

    return embedded










def _delay_embedding_plot(embedded):
    """Plot reconstructed attractor.

    The input must be obtained via `nk.delay_embedding()`
    """
    if embedded.shape[1] == 2:
        figure = _delay_embedding_plot_2D(embedded)
    else:
        figure = _delay_embedding_plot_3D(embedded)

    return figure

# =============================================================================
# Internal plots
# =============================================================================
def _delay_embedding_plot_2D(embedded):
    figure = plt.plot(embedded[:,0], embedded[:,1])
    return figure


def _delay_embedding_plot_3D(embedded):
    figure = plt.figure()
    axes = mpl_toolkits.mplot3d.Axes3D(figure)
    axes.plot3D(embedded[:,0], embedded[:,1], embedded[:,2])
    figure.add_axes(axes)
    return figure