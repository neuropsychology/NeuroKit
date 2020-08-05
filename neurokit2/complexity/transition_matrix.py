# -*- coding: utf-8 -*-
import itertools
import numpy as np
import pandas as pd
import scipy.stats


def transition_matrix(sequence):
    """Empirical transition matrix

    Also known as discrete Markov chains. Computes the observed transition matrix and performs a
    Chi-square test against the expected transition matrix.

    Based on https://github.com/Frederic-vW/eeg_microstates and https://github.com/maximtrp/mchmm

    Parameters
    ----------
    sequence : np.ndarray
        1D array of numbers.

    Returns
    -------
    dict
        Contains information of the expected and observed transition matrix.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> sequence = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 0, 0])
    >>> out = nk.transition_matrix(sequence)
    >>> out["Observed"] #doctest: +ELLIPSIS
              0         1         2
    0  0.750000  0.250000  0.000000
    1  0.333333  0.333333  0.333333
    2  0.000000  0.250000  0.750000

    """
    out = {}

    # Observed transtion matrix
    out["Observed"] = _transition_matrix_observed(sequence)

    # Expect transition matrix (theorethical)
    out["Expected"] = _transition_matrix_expected(out["Observed"])

    # Test against theorethical transitions
    results = scipy.stats.chisquare(f_obs=out["Observed"], f_exp=out["Expected"], axis=None)
    out["Transition_Chisq"] = results[0]
    out["Transition_df"] = len(out["Observed"])*(len(out["Observed"])-1)/2
    out["Transition_p"] = results[1]

    # Symmetry test
    out.update(_transition_matrix_symmetry(sequence))

    return out


def transition_matrix_simulate(matrix, n=10):
    """Markov chain simulation

    The algorithm is based on `scipy.stats.multinomial`. The code is heavily inspired by
    https://github.com/maximtrp/mchmm.


    Examples
    --------
    >>> import neurokit2 as nk
    >>> import numpy as np
    >>>
    >>> sequence = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 0, 0])
    >>> matrix = nk.transition_matrix(sequence)["Observed"]
    >>>
    >>> x = nk.transition_matrix_simulate(matrix, n=10)
    >>> x #doctest: +SKIP
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
def _transition_matrix_observed(sequence):
    """Empirical transition matrix

    Based on https://github.com/Frederic-vW/eeg_microstates and https://github.com/maximtrp/mchmm
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

    # filling in a row containing zeros with uniform p values
    uniform_p = 1 / n_states
    zero_row = np.argwhere(matrix.sum(axis=1) == 0).ravel()
    matrix[zero_row, :] = uniform_p

    # Convert to DataFrame
    out = pd.DataFrame(matrix, index=states, columns=states)
    return out


def _transition_matrix_expected(observed_matrix):

    expected_matrix = scipy.stats.contingency.expected_freq(observed_matrix.values)
    expected_matrix = pd.DataFrame(expected_matrix, index=observed_matrix.index, columns=observed_matrix.columns)
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

    for t in range(n-1):
        i = sequence[t]
        j = sequence[t+1]
        f_ij[states == i, states == j] += 1.0

    T = 0.0
    for i, j in np.ndindex(f_ij.shape):
        if (i != j):
            f = f_ij[i, j] * f_ij[j, i]
            if (f > 0):
                T += (f_ij[i, j] * np.log((2. * f_ij[i, j]) / (f_ij[i, j] + f_ij[j, i])))

    out = {}
    out["Symmetry_t"] = T * 2.0
    out["Symmetry_df"] = n_states*(n_states-1)/2
    out["Symmetry_p"] = scipy.stats.chi2.sf(out["Symmetry_t"], out["Symmetry_df"], loc=0, scale=1)
    return out



def _transition_matrix_stationarity(sequence, size=100):
    """Test conditional homogeneity of non-overlapping blocks of
    length l of symbolic sequence X with ns symbols
    cf. Kullback, Technometrics (1962), Table 9.1.

    ased on https://github.com/Frederic-vW/eeg_microstates
    """
    states = np.unique(sequence)
    n_states = len(states)
    n = len(sequence)
    r = int(np.floor(n / size))  # number of blocks
    if r < 5:
        raise ValueError(
            "NeuroKit error: _transition_matrix_stationarity(): the size of the blocks is too high.",
            " Decrease the 'size' argument.")

#    nl =  r* size

    f_ijk = np.zeros((r, n_states, n_states))
    f_ij = np.zeros((r, n_states))
    f_jk = np.zeros((n_states, n_states))
    f_i = np.zeros(r)
    f_j = np.zeros(n_states)

    # calculate f_ijk (time / block dep. transition matrix)
    for i in range(r):  # block index
        for ii in range(size-1):  # pos. inside the current block
            j = sequence[i*size + ii]
            k = sequence[i*size + ii + 1]
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
        if (f > 0):
            T += (f_ijk[i, j, k] * np.log((f_ijk[i, j, k] * f_j[j]) / (f_ij[i, j] * f_jk[j, k])))

    out = {}
    out["Stationarity_t"] = T * 2.0
    out["Stationarity_df"] = (r-1)*(n_states-1)*n_states
    out["Stationarity_p"] = scipy.stats.chi2.sf(out["Stationarity_t"], out["Stationarity_df"], loc=0, scale=1)
    return out
