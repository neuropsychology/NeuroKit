# -*- coding: utf-8 -*-
import numpy as np

from ..misc import check_random_state
from .transition_matrix import _sanitize_tm_input


def markov_simulate(tm, n=10, random_state=None):
    """**Markov Chain Simulation**

    Given a :func:`transition_matrix`, this function simulates the corresponding sequence of states
    (also known as a discrete Markov chain).

    Parameters
    ----------
    tm : pd.DataFrame
        A probability matrix obtained from :func:`transition_matrix`.
    n : int
        Length of the simulated sequence.
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator. See for ``misc.check_random_state`` for further information.

    Returns
    -------
    np.ndarray
        Sequence of states.

    See Also
    --------
    transition_matrix

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      sequence = [0, 0, 1, 2, 2, 2, 1, 0, 0, 3]
      tm, _ = nk.transition_matrix(sequence)

      x = nk.markov_simulate(tm, n=15)
      x

    """
    # Sanitize input
    tm = _sanitize_tm_input(tm)
    states = tm.columns.values

    # Start selection
    _start = np.argmax(tm.sum(axis=1) / tm.sum())

    # simulated sequence init
    seq = np.zeros(n, dtype=int)
    seq[0] = _start

    # Seed the random generator for reproducible results
    rng = check_random_state(random_state)

    # simulation procedure
    for i in range(1, n):
        _ps = tm.values[seq[i - 1]]
        _sample = rng.choice(len(_ps), p=_ps)
        seq[i] = _sample

    return states[seq]
