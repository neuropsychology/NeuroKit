# -*- coding: utf-8 -*-
import matplotlib.animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np


def complexity_embedding(signal, delay=1, dimension=3, show=False):
    """
    Time-delay embedding of a time series (a signal)

    A dynamical system can be described by a vector of numbers, called its 'state', that aims to provide a complete description of the system at some point in time. The set of all possible states is called the 'state space'.

    Takens's (1981) embedding theorem suggests that a sequence of measurements of a dynamic system includes in itself all the information required to completely reconstruct the state space. Delay coordinate embedding attempts to identify the state s of the system at some time t by searching the past history of observations for similar states, and, by studying the evolution of similar states, infer information about the future of the system.

    How to visualize the dynamics of a system? A sequence of state values over time is called a trajectory. Depending on the system, different trajectories can evolve to a common subset of state space called an attractor. The presence and behavior of attractors gives intuition about the underlying dynamical system. We can visualize the system and its attractors by plotting the trajectory of many different initial state values and numerically integrating them to approximate their continuous time evolution on discrete computers.

    This function is adapted from `EntroPy <https://github.com/raphaelvallat/entropy>`_ and is equivalent to
    the `delay_embedding()` function from 'nolds'.

    Parameters
    ----------
    signal : list, array or Series
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically 2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    show : bool
        Plot the reconstructed attractor.

    Returns
    -------
    array
        Embedded time-series, of shape (n_times - (order - 1) * delay, order)

    See Also
    ------------
    embedding_delay, embedding_dimension

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> # Artifical example
    >>> signal = nk.signal_simulate(duration=2, frequency=5, noise=0.01)
    >>>
    >>> embedded = nk.complexity_embedding(signal, delay=50, dimension=2, show=True)
    >>> embedded = nk.complexity_embedding(signal, delay=50, dimension=3, show=True)
    >>> embedded = nk.complexity_embedding(signal, delay=50, dimension=4, show=True)
    >>>
    >>> # Realistic example
    >>> ecg = nk.ecg_simulate(duration=60*4, sampling_rate=200)
    >>> signal = nk.ecg_rate(nk.ecg_peaks(ecg, sampling_rate=200)[0], sampling_rate=200)
    >>>
    >>> embedded = nk.complexity_embedding(signal, delay=250, dimension=2, show=True)
    >>> embedded = nk.complexity_embedding(signal, delay=250, dimension=3, show=True)
    >>> embedded = nk.complexity_embedding(signal, delay=250, dimension=4, show=True)

    References
    -----------
    - Gautama, T., Mandic, D. P., & Van Hulle, M. M. (2003, April). A differential entropy based method for determining the optimal embedding parameters of a signal. In 2003 IEEE International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings.(ICASSP'03). (Vol. 6, pp. VI-29). IEEE.

    """
    N = len(signal)

    # Sanity checks
    if dimension * delay > N:
        raise ValueError(
            "NeuroKit error: complexity_embedding(): dimension * delay should be lower than length of signal."
        )
    if delay < 1:
        raise ValueError("NeuroKit error: complexity_embedding(): 'delay' has to be at least 1.")

    Y = np.zeros((dimension, N - (dimension - 1) * delay))
    for i in range(dimension):
        Y[i] = signal[i * delay : i * delay + Y.shape[1]]
    embedded = Y.T

    if show is True:
        _embedding_plot(embedded)

    return embedded


# =============================================================================
# Internals
# =============================================================================


def _embedding_plot(embedded):
    """
    Plot reconstructed attractor.

    The input for this function must be obtained via `nk.complexity_embedding()`

    """
    if embedded.shape[1] == 2:
        figure = _embedding_plot_2D(embedded)
    elif embedded.shape[1] == 3:
        figure = _embedding_plot_3D(embedded)
    else:
        figure = _embedding_plot_4D(embedded)

    return figure


# =============================================================================
# Internal plots
# =============================================================================


def _embedding_plot_2D(embedded):
    return plt.plot(embedded[:, 0], embedded[:, 1], color="#3F51B5")


def _embedding_plot_3D(embedded):
    return _plot_3D_colored(x=embedded[:, 0], y=embedded[:, 1], z=embedded[:, 2], color=embedded[:, 2], rotate=False)


def _embedding_plot_4D(embedded):
    return _plot_3D_colored(x=embedded[:, 0], y=embedded[:, 1], z=embedded[:, 2], color=embedded[:, 3], rotate=False)


# =============================================================================
# Plotting
# =============================================================================
def _plot_3D_colored(x, y, z, color=None, rotate=False):
    if color is None:
        color = z

    # Create a set of line segments
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color
    norm = plt.Normalize(color.min(), color.max())
    cmap = plt.get_cmap("plasma")
    colors = cmap(norm(color))

    # Plot
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    for i in range(len(x) - 1):
        seg = segments[i]
        (l,) = ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=colors[i])
        l.set_solid_capstyle("round")

    if rotate is True:
        fig = _plot_3D_colored_rotate(fig, ax)

    return fig


def _plot_3D_colored_rotate(fig, ax):
    def rotate(angle):
        ax.view_init(azim=angle)

    fig = matplotlib.animation.FuncAnimation(
        fig, rotate, frames=np.arange(0, 361, 1), interval=10, cache_frame_data=False
    )

    return fig
