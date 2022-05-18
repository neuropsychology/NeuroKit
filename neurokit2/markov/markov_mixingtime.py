# -*- coding: utf-8 -*-
import numpy as np

from .transition_matrix import _sanitize_tm_input


def markov_mixingtime(tm):
    """**Markov Chain Mixing Time**

    The Mixing time (also known as relaxation time) is the inverse of spectral gap, which is the
    difference between the two largest eigenvalues of the transition matrix. The Mixing time of a
    Markov chain tells us how long does it take for a run to go near the stationary distribution
    (for convergence to happen).

    Parameters
    ----------
    tm : pd.DataFrame
        A transition matrix obtained from :func:`transition_matrix`.

    Returns
    -------
    float
        Mixing time of the Markov chain.

    See Also
    --------
    transition_matrix

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      sequence = [0, 0, 1, 2, 2, 2, 1, 0, 0, 3]

      tm, _ = nk.transition_matrix(sequence)

      nk.markov_mixingtime(tm)

    References
    -----------
    * Levin, D. A., & Peres, Y. (2017). Markov chains and mixing times (Vol. 107). American
      Mathematical Society.

    """
    # Sanitize input
    tm = _sanitize_tm_input(tm)

    ev = np.linalg.eigvals(tm)
    ev = np.real(ev)

    # ascending
    ev.sort()
    # Spectral gap = Largest (last) - second largest
    sg = ev[-1] - ev[-2]

    # mixing time (aka, relaxation time)
    return 1.0 / sg
