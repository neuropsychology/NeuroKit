# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np

from ..misc import NeuroKitWarning
from .complexity_attractor import _attractor_equation, complexity_attractor


def complexity_embedding(signal, delay=1, dimension=3, show=False, **kwargs):
    """Time-delay embedding of a signal

    A dynamical system can be described by a vector of numbers, called its 'state', that aims to provide
    a complete description of the system at some point in time. The set of all possible states is called
    the 'state space'.

    Takens's (1981) embedding theorem suggests that a sequence of measurements of a dynamic system includes
    in itself all the information required to completely reconstruct the state space. Delay coordinate
    embedding attempts to identify the state s of the system at some time t by searching the past history
    of observations for similar states, and, by studying the evolution of similar states, infer information
    about the future of the system.

    How to visualize the dynamics of a system? A sequence of state values over time is called a trajectory.
    Depending on the system, different trajectories can evolve to a common subset of state space called
    an attractor. The presence and behavior of attractors gives intuition about the underlying dynamical
    system. We can visualize the system and its attractors by plotting the trajectory of many different
    initial state values and numerically integrating them to approximate their continuous time evolution
    on discrete computers.

    This function is adapted from `EntroPy <https://github.com/raphaelvallat/entropy>`_ and is equivalent
    to the `delay_embedding()` function from 'nolds'.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values. Can also be a string, such as
        ``"lorenz"`` (Lorenz attractor), ``"rossler"`` (Rössler attractor), or ``"clifford"`` (Clifford attractor)
        to obtain a pre-defined attractor.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common
        to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or
        to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns
        an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    show : bool
        Plot the reconstructed attractor.
    **kwargs
        Other arguments to be passed to the plotting of the attractor (see ``complexity_attractor()``).

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
    >>> # Basic example
    >>> signal = [1, 2, 3, 2.5, 2.0, 1.5]
    >>> embedded = nk.complexity_embedding(signal, delay = 2, dimension = 2, show=True) #doctest: +SKIP
    >>>
    >>> # Artifical example
    >>> signal = nk.signal_simulate(duration=2, frequency=5, noise=0.01)
    >>>
    >>> embedded = nk.complexity_embedding(signal, delay=50, dimension=2, show=True) #doctest: +SKIP
    >>> embedded = nk.complexity_embedding(signal, delay=50, dimension=3, show=True) #doctest: +SKIP
    >>> embedded = nk.complexity_embedding(signal, delay=50, dimension=4, show=True) #doctest: +SKIP
    >>>
    >>> # Realistic example
    >>> ecg = nk.ecg_simulate(duration=60*4, sampling_rate=200)
    >>> signal = nk.ecg_rate(nk.ecg_peaks(ecg, sampling_rate=200)[0], sampling_rate=200, desired_length=len(ecg))
    >>>
    >>> embedded = nk.complexity_embedding(signal, delay=250, dimension=2, show=True) #doctest: +SKIP
    >>> embedded = nk.complexity_embedding(signal, delay=250, dimension=3, show=True) #doctest: +SKIP
    >>> embedded = nk.complexity_embedding(signal, delay=250, dimension=4, show=True) #doctest: +SKIP

    References
    -----------
    - Gautama, T., Mandic, D. P., & Van Hulle, M. M. (2003, April). A differential entropy based method
      for determining the optimal embedding parameters of a signal. In 2003 IEEE International Conference
      on Acoustics, Speech, and Signal Processing, 2003. Proceedings.(ICASSP'03). (Vol. 6, pp. VI-29). IEEE.

    """
    # If string
    if isinstance(signal, str):
        return _attractor_equation(signal, **kwargs)

    N = len(signal)

    # Sanity checks
    if isinstance(delay, float):
        warn("`delay` must be an integer. Running `int(delay)`", category=NeuroKitWarning)
        delay = int(delay)
    if isinstance(dimension, float):
        warn("`dimension` must be an integer. Running `int(dimension)`", category=NeuroKitWarning)
        dimension = int(dimension)
    if dimension * delay > N:
        raise ValueError(
            "NeuroKit error: complexity_embedding(): dimension * delay should be lower than",
            " the length of the signal.",
        )
    if delay < 1:
        raise ValueError("NeuroKit error: complexity_embedding(): 'delay' has to be at least 1.")

    Y = np.zeros((dimension, N - (dimension - 1) * delay))
    for i in range(dimension):
        Y[i] = signal[i * delay : i * delay + Y.shape[1]]
    embedded = Y.T

    if show is True:
        complexity_attractor(embedded, **kwargs)

    return embedded
