# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import itertools
import scipy.stats





def transition_matrix(sequence):
    """Empirical transition matrix

    Also known as discrete Markov chains. Computes the observed transition matrix and performs a
    Chi-square test against the expected transition matrix.

    Based on https://github.com/Frederic-vW/eeg_microstates and https://github.com/maximtrp/mchmm

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> sequence = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 0, 0])
    >>> out = nk.transition_matrix(sequence)
    >>> out["Observed"]
    """
    out = {}

    # Observed transtion matrix
    out["Observed"] = _transition_matrix_observed(sequence)

    # Expect transition matrix (theorethical)
    out["Expected"] = _transition_matrix_expected(out["Observed"])

    # Test against random transitions
    results = scipy.stats.chisquare(f_obs=out["Observed"], f_exp=out["Expected"], axis=None)
    out["Chisq"] = results[0]
    out["p"] = results[1]

    return out






def transition_matrix_simulate(matrix, n=10):
    """Markov chain simulation

    The algorithm is based on `scipy.stats.multinomial`. The code is heavily inspired by
    https://github.com/maximtrp/mchmm.


    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> sequence = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 0, 0])
    >>> matrix = nk.transition_matrix(sequence)["Observed"]
    >>>
    >>> nk.transition_matrix_simulate(matrix, n=10)
    """
    states = matrix.columns.values

    # Start selection
    _start = np.argmax(matrix.sum(axis=1) / matrix.sum())

    # simulated sequence init
    seq = np.zeros(n, dtype=np.int)
    seq[0] = _start

    # random seeds
    random_states = np.random.randint(0, n, n)

    # simulation procedure
    for i in range(1, n):
        _ps = matrix.values[seq[i-1]]
        _sample = np.argmax(scipy.stats.multinomial.rvs(1, _ps, 1, random_state=random_states[i]))
        seq[i] = _sample

    return states[seq]




#def transition_matrix_plot(matrix):
#    """
#    """
#
#    def _get_markov_edges(matrix):
#        edges = {}
#        for col in matrix.columns:
#            for idx in matrix.index:
#                edges[(idx,col)] = matrix.loc[idx,col]
#        return edges
#
#
#    import networkx as nx
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




# =============================================================================
# Internals
# =============================================================================
def _transition_matrix_observed(sequence):
    """Empirical transition matrix

    Based on https://github.com/Frederic-vW/eeg_microstates and https://github.com/maximtrp/mchmm

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> sequence = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 0, 0])
    >>> _transition_matrix_observed(sequence)  #doctest: +SKIP
    """
    states = np.unique(sequence)
    n_states = len(states)

    # Get observed transition matrix
    matrix = np.zeros((n_states, n_states))
    for x, y in itertools.product(range(n_states), repeat=2):
        xi = np.argwhere(sequence == states[x]).flatten()
        yi = xi + 1
        yi = yi[yi < len(sequence)]
        matrix[x, y] = np.count_nonzero(sequence[yi] == states[y])

    # Convert to probabilities
    matrix = matrix / np.sum(matrix, axis=0)[:, None]

    # Convert to DataFrame
    out = pd.DataFrame(matrix, index=states, columns=states)
    return out


def _transition_matrix_expected(observed_matrix):
    """
    """
    expected_matrix = scipy.stats.contingency.expected_freq(observed_matrix.values)
    expected_matrix = pd.DataFrame(expected_matrix, index=observed_matrix.index, columns=observed_matrix.columns)
    return expected_matrix



