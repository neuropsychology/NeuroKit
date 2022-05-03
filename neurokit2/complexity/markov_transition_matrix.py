# -*- coding: utf-8 -*-
import itertools

import numpy as np
import pandas as pd
import scipy.stats

from ..misc import as_vector


def transition_matrix(sequence):
    """**Transition Matrix**

    A Transition Matrix (also known as a stochastic matrix or a Markov matrix) is the first step to
    describe a sequence of states, also known as **discrete Markov chains**. Each of its entries is
    a probability of transitioning from one state to the other.

    Computes the observed transition matrix and performs a
    Chi-square test against the expected transition matrix.

    Parameters
    ----------
    sequence : Union[list, np.array, pd.Series]
        A list of discrete states.

    See Also
    --------
    markov_simulate, markov_test_random

    Returns
    -------
    pd.DataFrame
        The empirical (observed) transition matrix.
    dict
        A dictionnary containing additional information.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      sequence = [0, 0, 1, 2, 2, 2, 1, 0, 0, 3]
      tm, _ = nk.transition_matrix(sequence)
      tm

    """
    out = {}

    sequence = as_vector(sequence)

    # Observed transition matrix
    states = np.unique(sequence)
    n_states = len(states)

    # Get observed transition matrix
    freqs = np.zeros((n_states, n_states))
    for x, y in itertools.product(range(n_states), repeat=2):
        xi = np.argwhere(sequence == states[x]).flatten()
        yi = xi + 1
        yi = yi[yi < len(sequence)]
        freqs[x, y] = np.count_nonzero(sequence[yi] == states[y])

    # Convert to probabilities
    tm = freqs / np.sum(freqs, axis=0)[:, None]

    # filling in a row containing zeros with uniform p values
    uniform_p = 1 / n_states
    zero_row = np.argwhere(tm.sum(axis=1) == 0).ravel()
    tm[zero_row, :] = uniform_p

    # Convert to DataFrame
    tm = pd.DataFrame(tm, index=states, columns=states)
    freqs = pd.DataFrame(freqs, index=states, columns=states)

    # # Expect transition matrix (theoretical)
    # out["Expected"] = _transition_matrix_expected(tm)

    # # Test against theoretical transitions
    # results = scipy.stats.chisquare(f_obs=tm, f_exp=out["Expected"], axis=None)
    # out["Transition_Chisq"] = results[0]
    # out["Transition_df"] = len(tm) * (len(tm) - 1) / 2
    # out["Transition_p"] = results[1]

    # # Symmetry test
    # out.update(_transition_matrix_symmetry(sequence))

    return tm, {"Occurrences": freqs}


def _sanitize_tm_input(tm, probs=True):
    # If symmetric dataframe, then surely a transition matrix
    if isinstance(tm, pd.DataFrame) and tm.shape[1] == tm.shape[0]:
        if tm.values.max() > 1:
            if probs is True:
                raise ValueError(
                    "Transition matrix must be a probability matrix (all probabilities must be"
                    " < 1)."
                )
            else:
                return tm
        else:
            if probs is True:
                return tm
            else:
                raise ValueError(
                    "Transition matrix must be a frequency matrix containing counts and not"
                    " probabilities. Please pass the `info['Occurrences']` object instead of"
                    " the transition matrix."
                )

    # Otherwise, conver to TM
    else:
        return transition_matrix(tm)


# def transition_matrix_plot(matrix):
#    """
#    """
#    print("Sorry, we didn't find a statisfactory way of plotting the transition graphs. Consider ",
#          "helping if you have some plotting skills!")
#
#    try:
#        import networkx as nx
#    except ImportError:
#        raise ImportError(
#            "NeuroKit error: transition_matrix_plot(): the 'networkx' module is required for this ",
#            "function to run. Please install it first (`pip install networkx`).",
#        )
#
#    def _get_markov_edges(matrix):
#        edges = {}
#        for col in matrix.columns:
#            for idx in matrix.index:
#                edges[(idx,col)] = matrix.loc[idx,col]
#        return edges
#
#    states = matrix.columns.values
#
#    # create graph object
#    G = nx.MultiDiGraph()
#
#    # nodes correspond to states
#    G.add_nodes_from(states)
#
#    # edges represent transition probabilities
#    for k, v in _get_markov_edges(matrix).items():
#        tmp_origin, tmp_destination = k[0], k[1]
#        G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
#
#    # Create edge labels for jupyter plot but is not necessary
#    edge_labels = {(n1,n2):"%.2f" %d['label'] for n1, n2, d in G.edges(data=True)}
#
#    pos = nx.spring_layout(G)
#    pos = nx.circular_layout(G)
#    nx.draw_networkx_edges(G , pos=pos, edge_color="grey")
#    nx.draw_networkx_edge_labels(G , pos=pos, edge_labels=edge_labels, clip_on=False)
#
#    nx.draw_networkx_nodes(G, pos=pos, node_color="red")
#    nx.draw_networkx_labels(G , pos=pos)
#
#    nx.drawing.nx_pydot.to_pydot(G, 'markov.dot')
#    A = nx.nx_agraph.to_agraph(G)


# =============================================================================
# Internals
# =============================================================================
def _transition_matrix_expected(observed_matrix):

    expected_matrix = scipy.stats.contingency.expected_freq(observed_matrix.values)
    expected_matrix = pd.DataFrame(
        expected_matrix, index=observed_matrix.index, columns=observed_matrix.columns
    )
    return expected_matrix


def _transition_matrix_symmetry(sequence):
    """Symmetry Test

    If significant, then then transition matrix is considered as asymmetric.

    Based on https://github.com/Frederic-vW/eeg_microstates
    """
    states = np.unique(sequence)
    n_states = len(states)
    n = len(sequence)
    f_ij = np.zeros((n_states, n_states))

    for t in range(n - 1):
        i = sequence[t]
        j = sequence[t + 1]
        f_ij[states == i, states == j] += 1.0

    T = 0.0
    for i, j in np.ndindex(f_ij.shape):
        if i != j:
            f = f_ij[i, j] * f_ij[j, i]
            if f > 0:
                T += f_ij[i, j] * np.log((2.0 * f_ij[i, j]) / (f_ij[i, j] + f_ij[j, i]))

    out = {}
    out["Symmetry_t"] = T * 2.0
    out["Symmetry_df"] = n_states * (n_states - 1) / 2
    out["Symmetry_p"] = scipy.stats.chi2.sf(out["Symmetry_t"], out["Symmetry_df"], loc=0, scale=1)
    return out


def _transition_matrix_stationarity(sequence, size=100):
    """Test conditional homogeneity of non-overlapping blocks of
    length l of symbolic sequence X with ns symbols
    cf. Kullback, Technometrics (1962), Table 9.1.

    based on https://github.com/Frederic-vW/eeg_microstates
    """
    states = np.unique(sequence)
    n_states = len(states)
    n = len(sequence)
    r = int(np.floor(n / size))  # number of blocks
    if r < 5:
        raise ValueError(
            "NeuroKit error: _transition_matrix_stationarity(): the size of the blocks is too high.",
            " Decrease the 'size' argument.",
        )

    #    nl =  r* size

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

    out = {}
    out["Stationarity_t"] = T * 2.0
    out["Stationarity_df"] = (r - 1) * (n_states - 1) * n_states
    out["Stationarity_p"] = scipy.stats.chi2.sf(
        out["Stationarity_t"], out["Stationarity_df"], loc=0, scale=1
    )
    return out
