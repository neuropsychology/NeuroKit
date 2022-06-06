import numpy as np
import pandas as pd
import scipy.stats

from .transition_matrix import transition_matrix


def markov_test_markovity(sequence):
    """**Test of Markovity**

    The Markovity (also known as Markovianity) of a symbolic sequence.

    .. note::

      We would like to extend this to different orders (order 1, 2), but we lack the skills. If you
      are interested, please get in touch!

    Parameters
    ----------
    sequence : Union[list, np.array, pd.Series]
        A list of discrete states.

    See Also
    --------
    markov_test_random, markov_test_symmetry

    Returns
    -------
    dict
        Contains indices of the test.


    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      sequence = [0, 0, 1, 2, 2, 2, 1, 0, 0, 3, 1]

      nk.markov_test_markovity(sequence)

    References
    ----------
    * Kullback, S., Kupperman, M., & Ku, H. H. (1962). Tests for contingency tables and Markov
      chains. Technometrics, 4(4), 573-608.

    """
    _, info = transition_matrix(sequence)

    # Extract frequency matrix
    if isinstance(info["Occurrences"], pd.DataFrame):
        fm = info["Occurrences"].values
    else:
        fm = info["Occurrences"]
    k = len(fm)
    valid = fm != 0

    # Get numerator and denominator
    num = fm * len(sequence)
    den = np.tile(fm.sum(axis=1), (k, 1)).T * np.tile(fm.sum(axis=0), (k, 1))

    # Compute statistics
    out = {
        "Markovity_t": 2 * np.sum(fm[valid] * np.log(num[valid] / den[valid])),
        "Markovity_df": (k - 1.0) * (k - 1.0),
    }

    # Chi2 test
    out["Markovity_p"] = scipy.stats.chi2.sf(
        out["Markovity_t"],
        out["Markovity_df"],
        loc=0,
        scale=1,
    )
    return out


# def testMarkov1(sequence, verbose=True):
#     """Test first-order Markovianity of symbolic sequence X with ns symbols.
#     Null hypothesis:
#     first-order MC <=>
#     p(X[t+1] | X[t]) = p(X[t+1] | X[t], X[t-1])
#     cf. Kullback, Technometrics (1962), Tables 8.1, 8.2, 8.6.
#     Args:
#         x: symbolic sequence, symbols = [0, 1, 2, ...]
#         ns: number of symbols
#         alpha: significance level
#     Returns:
#         p: p-value of the Chi2 test for independence
#     """
#     sequence = [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]
#     _, info = nk.transition_matrix(sequence, order=2)
#     fm = info["Occurrences"]

#     X = sequence
#     ns = len(np.unique(X))
#     n = len(X)
#     f_ijk = np.zeros((ns, ns, ns))
#     f_ij = np.zeros((ns, ns))
#     f_jk = np.zeros((ns, ns))
#     f_j = np.zeros(ns)
#     for t in range(n - 2):
#         i = X[t]
#         j = X[t + 1]
#         k = X[t + 2]
#         f_ijk[i, j, k] += 1.0
#         f_ij[i, j] += 1.0
#         f_jk[j, k] += 1.0
#         f_j[j] += 1.0
#     T = 0.0
#     for i, j, k in np.ndindex(f_ijk.shape):
#         f = f_ijk[i][j][k] * f_j[j] * f_ij[i][j] * f_jk[j][k]
#         if f > 0:
#             num_ = f_ijk[i, j, k] * f_j[j]
#             print(num_)
#             den_ = f_ij[i, j] * f_jk[j, k]
#             T += f_ijk[i, j, k] * np.log(num_ / den_)
#     T *= 2.0
#     df = ns * (ns - 1) * (ns - 1)
#     # p = chi2test(T, df, alpha)
#     p = scipy.stats.chi2.sf(T, df, loc=0, scale=1)
#     if verbose:
#         print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")
#     return p
