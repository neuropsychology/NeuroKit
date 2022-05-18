# -*- coding: utf-8 -*-
import pandas as pd
import scipy.stats

from .transition_matrix import _sanitize_tm_input


def markov_test_random(fm):
    """**Is the Markov process random?**

    This function computes the expected (theoretical) transition matrix if the order of appearance
    of each state was governed only by their overall prevalence, and that a previous state had no
    influence on the next state. This "random" matrix is then compered again the observed one, and
    a Chi2 test is conducted.

    If significant (e.g., ``*p*-value < .05``), one can reject the hypothesis that observed Markov
    process is random, and conclude that past states have an influence on next states.

    Parameters
    ----------
    fm : pd.DataFrame
        A frequency matrix obtained from :func:`transition_matrix`.

    Returns
    -------
    dict
        Contains indices of the Chi2 test.

    See Also
    --------
    transition_matrix

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      sequence = [0, 0, 1, 2, 2, 2, 1, 0, 0, 3]
      _, info = nk.transition_matrix(sequence)

      result = nk.markov_test_random(info["Occurrences"])
      result["Random_p"]

    """
    # Sanitize input
    fm = _sanitize_tm_input(fm, probs=False)

    # Remove rows with no occurence
    fm = fm.loc[~(fm.sum(axis=1) == 0).values]

    out = {}

    # Expect transition matrix (theoretical)
    out["Random_Matrix"] = scipy.stats.contingency.expected_freq(fm.values)
    out["Random_Matrix"] = pd.DataFrame(out["Random_Matrix"], index=fm.index, columns=fm.columns)

    # Chi-square test
    results = scipy.stats.chisquare(f_obs=fm, f_exp=out["Random_Matrix"], axis=None)

    # Store results
    out["Random_Chi2"] = results[0]
    out["Random_df"] = len(fm) * (len(fm) - 1) / 2
    out["Random_p"] = results[1]

    return out
