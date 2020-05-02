# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


def delay_embedding(signal, delay=1, dimension=3, show=False):
    """Time-delay embedding of a time series (a signal)

    A dynamical system can be described by a vector of numbers, called its 'state', that aims to provide a complete description of the system at some point in time. The set of all possible states is called the 'state space'.

    Takens's (1981) embedding theorem suggests that a sequence of measurements of a dynamic system includes in itself all the information required to completely reconstruct the state space. Delay coordinate embedding attempts to identify the state s of the system at some time t by searching the past history of observations for similar states, and, by studying the evolution of similar states, infer information about the future of the system.

    How to visualize the dynamics of a system? A sequence of state values over time is called a trajectory. Depending on the system, different trajectories can evolve to a common subset of state space called an attractor. The presence and behavior of attractors gives intuition about the underlying dynamical system. We can visualize the system and its attractors by plotting the trajectory of many different initial state values and numerically integrating them to approximate their continuous time evolution on discrete computers.

    This function is adapted from `EntroPy <https://github.com/raphaelvallat/entropy>`_ and is equivalent to
    the `delay_embedding()` function from 'nolds'.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    delay : int
        Time delay (Tau).
    order : int
        Embedding dimension (m), sometimes referred to as 'order'.
    show : bool
        Plot the reconstructed attractor.

    Returns
    -------
    array
        Embedded time-series, of shape (n_times - (order - 1) * delay, order)

    See Also
    ------------

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5, noise=0.01)
    >>>
    >>> embedded = nk.delay_embedding(signal, delay=50, dimension=2, show=True)
    >>> embedded = nk.delay_embedding(signal, delay=50, dimension=3, show=True)
    >>> embedded = nk.delay_embedding(signal, delay=50, dimension=4, show=True)
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

    The input for this function must be obtained via `nk.delay_embedding()`
    """
    if embedded.shape[1] == 2:
        figure = _delay_embedding_plot_2D(embedded)
    elif embedded.shape[1] == 3:
        figure = _delay_embedding_plot_3D(embedded)
    else:
        figure = _delay_embedding_plot_4D(embedded)

    return figure


# =============================================================================
# Internal plots
# =============================================================================

def _delay_embedding_plot_2D(embedded):
    figure = plt.plot(embedded[:,0], embedded[:,1], color='#3F51B5')
    return figure


def _delay_embedding_plot_3D(embedded):
    figure = _plot_3D_colored(x=embedded[:,0],
                              y=embedded[:,1],
                              z=embedded[:,2],
                              color=embedded[:,2])
    return figure

def _delay_embedding_plot_4D(embedded):
    figure = _plot_3D_colored(x=embedded[:,0],
                              y=embedded[:,1],
                              z=embedded[:,2],
                              color=embedded[:,3])
    return figure



# =============================================================================
# Plotting
# =============================================================================
def _plot_3D_colored(x, y, z, color=None):
    if color is None:
        color = z

    # Create a set of line segments
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color
    norm=plt.Normalize(color.min(), color.max())
    cmap = plt.get_cmap('plasma')
    colors = cmap(norm(color))

    # Plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in range(len(x)-1):
        seg = segments[i]
        l, = ax.plot(seg[:,0], seg[:,1], seg[:,2], color=colors[i])
        l.set_solid_capstyle('round')

    return fig