# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats

from .transition_matrix import _sanitize_tm_input


def markov_simulate(tm, n=10):
    """**Markov Chain Simulation**

    Given a :func:`transition_matrix`, this function simulates the corresponding sequence of states
    (also known as a discrete Markov chain).

    Parameters
    ----------
    tm : pd.DataFrame
        A probability matrix obtained from :func:`transition_matrix`.
    n : int
        Length of the simulated sequence.

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

    # random seeds
    random_states = np.random.randint(0, n, n)

    # simulation procedure
    for i in range(1, n):
        _ps = tm.values[seq[i - 1]]
        _sample = np.argmax(scipy.stats.multinomial.rvs(1, _ps, 1, random_state=random_states[i]))
        seq[i] = _sample

    return states[seq]
