# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np

from ..misc import NeuroKitWarning
from ..signal import signal_sanitize
from .utils_complexity_attractor import (_attractor_equation,
                                         complexity_attractor)


def complexity_embedding(signal, delay=1, dimension=3, show=False, **kwargs):
    """**Time-delay Embedding of a Signal**

    Time-delay embedding is one of the key concept of complexity science. It is based on the idea
    that a dynamical system can be described by a vector of numbers, called its *'state'*, that
    aims to provide a complete description of the system at some point in time. The set of all
    possible states is called the *'state space'*.

    Takens's (1981) embedding theorem suggests that a sequence of measurements of a dynamic system
    includes in itself all the information required to completely reconstruct the state space.
    Time-delay embedding attempts to identify the state *s* of the system at some time *t* by
    searching the past history of observations for similar states, and, by studying the evolution
    of similar states, infer information about the future of the system.

    **Attractors**

    How to visualize the dynamics of a system? A sequence of state values over time is called a
    trajectory. Depending on the system, different trajectories can evolve to a common subset of
    state space called an attractor. The presence and behavior of attractors gives intuition about
    the underlying dynamical system. We can visualize the system and its attractors by plotting the
    trajectory of many different initial state values and numerically integrating them to
    approximate their continuous time evolution on discrete computers.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values. Can also be a string,
        such as ``"lorenz"`` (Lorenz attractor), ``"rossler"`` (Rössler attractor), or
        ``"clifford"`` (Clifford attractor) to obtain a pre-defined attractor.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    show : bool
        Plot the reconstructed attractor. See :func:`complexity_attractor` for details.
    **kwargs
        Other arguments to be passed to :func:`complexity_attractor`.

    Returns
    -------
    array
        Embedded time-series, of shape ``length - (dimension - 1) * delay``

    See Also
    ------------
    complexity_delay, complexity_dimension, complexity_attractor

    Examples
    ---------
    **Example 1**: Understanding the output

    .. ipython

      import neurokit2 as nk

      # Basic example
      signal = [1, 2, 3, 2.5, 2.0, 1.5]
      embedded = nk.complexity_embedding(signal, delay = 2, dimension = 2)
      embedded

    The first columns contains the beginning of the signal, and the second column contains the
    values at *t+2*.

    **Example 2**: 2D, 3D, and "4D" Attractors. Note that 3D attractors are slow to plot.

    .. ipython

      # Artifical example
      signal = nk.signal_simulate(duration=4, sampling_rate=200, frequency=5, noise=0.01)

      @savefig p_complexity_embedding1.png scale=100%
      embedded = nk.complexity_embedding(signal, delay=50, dimension=2, show=True)
      @suppress
      plt.close()

    .. ipython

      @savefig p_complexity_embedding2.png scale=100%
      embedded = nk.complexity_embedding(signal, delay=50, dimension=3, show=True)
      @suppress
      plt.close()

    .. ipython

      @savefig p_complexity_embedding3.png scale=100%
      embedded = nk.complexity_embedding(signal, delay=50, dimension=4, show=True)
      @suppress
      plt.close()

    In the last 3D-attractor, the 4th dimension is represented by the color.

    **Example 3**: Attractor of heart rate

      ecg = nk.ecg_simulate(duration=60*4, sampling_rate=200)
      peaks, _ = nk.ecg_peaks(ecg, sampling_rate=200)
      signal = nk.ecg_rate(peaks, sampling_rate=200, desired_length=len(ecg))

      @savefig p_complexity_embedding4.png scale=100%
      embedded = nk.complexity_embedding(signal, delay=250, dimension=2, show=True)
      @suppress
      plt.close()

    References
    -----------
    * Gautama, T., Mandic, D. P., & Van Hulle, M. M. (2003, April). A differential entropy based
      method for determining the optimal embedding parameters of a signal. In 2003 IEEE
      International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings.
      (ICASSP'03). (Vol. 6, pp. VI-29). IEEE.
    * Takens, F. (1981). Detecting strange attractors in turbulence. In Dynamical systems and
      turbulence, Warwick 1980 (pp. 366-381). Springer, Berlin, Heidelberg.

    """
    # If string
    if isinstance(signal, str):
        return _attractor_equation(signal, **kwargs)

    N = len(signal)
    signal = signal_sanitize(signal)

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
