# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats

from .transition_matrix import _sanitize_tm_input


def markov_test_symmetry(fm):
    """**Is the Markov process symmetric?**

    Performs a symmetry test, to test if for instance if the transitions A -> B and B -> A occur
    with the same probability. If significant (e.g., ``*p*-value < .05``), one can reject the
    hypothesis that observed Markov process is symmetric, and conclude that it the transition
    matrix is asymmetric.

    Parameters
    ----------
    fm : pd.DataFrame
        A frequency matrix obtained from :func:`transition_matrix`.

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
      _, info = nk.transition_matrix(sequence)

      result = nk.markov_test_symmetry(info["Occurrences"])
      result["Symmetry_p"]

    References
    ----------
    * Kullback, S., Kupperman, M., & Ku, H. H. (1962). Tests for contingency tables and Markov
      chains. Technometrics, 4(4), 573-608.

    """
    # Sanitize input
    fm = _sanitize_tm_input(fm, probs=False)

    # Convert to array
    fm = fm.values

    # Start computation
    t = 0.0
    for i, j in np.ndindex(fm.shape):
        if i != j:
            f = fm[i, j] * fm[j, i]
            if f > 0:
                t += fm[i, j] * np.log((2.0 * fm[i, j]) / (fm[i, j] + fm[j, i]))

    # Run test
    out = {"Symmetry_t": t * 2.0, "Symmetry_df": len(fm) * (len(fm) - 1) / 2}
    out["Symmetry_p"] = scipy.stats.chi2.sf(out["Symmetry_t"], out["Symmetry_df"], loc=0, scale=1)
    return out
