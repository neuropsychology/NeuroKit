# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats


def markov_test_homogeneity(sequence, size=10):
    """**Is the Markov process homogeneous?**

    Performs a homogeneity test that tests the null hypothesis that the samples are
    homogeneous, i.e., from the same - but unspecified - population, against the alternative
    hypothesis that at least one pair of samples is from different populations.

    Parameters
    ----------
    sequence : Union[list, np.array, pd.Series]
        A list of discrete states.
    size : int
        The size of the non-overlapping windows to split the sequence.

    Returns
    -------
    dict
        Contains indices of the test.

    See Also
    --------
    transition_matrix

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      sequence = [0, 0, 1, 2, 2, 2, 1, 0, 0, 3]

      result = nk.markov_test_homogeneity(sequence, size=2)
      result["Homogeneity_p"]

    References
    ----------
    * Kullback, S., Kupperman, M., & Ku, H. H. (1962). Tests for contingency tables and Markov
      chains. Technometrics, 4(4), 573-608.

    """

    states = np.unique(sequence)
    n_states = len(states)
    n = len(sequence)
    r = int(np.floor(n / size))  # number of blocks
    if r < 5:
        raise ValueError("The size of the blocks is too high. Decrease the 'size' argument.")
    f_ijk = np.zeros((r, n_states, n_states))
    f_ij = np.zeros((r, n_states))
    f_jk = np.zeros((n_states, n_states))
    f_i = np.zeros(r)
    f_j = np.zeros(n_states)

    # calculate f_ijk (time / block dep. transition matrix)
    for i in range(r):  # block index
        for ii in range(size - 1):  # pos. inside the current block
            j = sequence[i * size + ii]
            k = sequence[i * size + ii + 1]
            f_ijk[i, j, k] += 1.0
            f_ij[i, j] += 1.0
            f_jk[j, k] += 1.0
            f_i[i] += 1.0
            f_j[j] += 1.0

    # conditional homogeneity (Markovianity stationarity)
    T = 0.0
    for i, j, k in np.ndindex(f_ijk.shape):
        # conditional homogeneity
        f = f_ijk[i, j, k] * f_j[j] * f_ij[i, j] * f_jk[j, k]
        if f > 0:
            T += f_ijk[i, j, k] * np.log((f_ijk[i, j, k] * f_j[j]) / (f_ij[i, j] * f_jk[j, k]))

    out = {"Homogeneity_t": T * 2.0, "Homogeneity_df": (r - 1) * (n_states - 1) * n_states}
    out["Homogeneity_p"] = scipy.stats.chi2.sf(
        out["Homogeneity_t"], out["Homogeneity_df"], loc=0, scale=1
    )
    return out
