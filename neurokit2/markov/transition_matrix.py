# -*- coding: utf-8 -*-
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import as_vector


def transition_matrix(sequence, show=False):
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
    show : bool
        Displays the transition matrix heatmap.

    See Also
    --------
    markov_simulate, markov_test_random, markov_test_symmetry

    Returns
    -------
    pd.DataFrame
        The empirical (observed) transition matrix.
    dict
        A dictionnary containing additional information, such as the Frequency Matrix (**fm**;
        accessible via the key ``"Occurrences"``), useful for some tests.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      sequence = [0, 0, 1, 2, 2, 2, 1, 0, 0, 3]

      @savefig p_transition_matrix1.png scale=100%
      tm, _ = nk.transition_matrix(sequence, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      tm

    """
    sequence = as_vector(sequence)

    # Observed transition matrix
    states = np.unique(sequence)
    n_states = len(states)

    # order=1
    # freq = np.zeros((n_states,) * (order + 1))
    # for _ind in zip(*[sequence[_x:] for _x in range(order + 1)]):
    #     freq[_ind] += 1
    # freq

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

    if show is True:
        fig, ax = plt.subplots()
        ax.imshow(tm, cmap="Reds", interpolation="nearest")
        ax.set_xticks(np.arange(len(tm)))
        ax.set_yticks(np.arange(len(tm)))
        ax.set_xticklabels(tm.columns)
        ax.set_yticklabels(tm.index)

        # Loop over data dimensions and create text annotations.
        for i, row in enumerate(tm.index):
            for j, col in enumerate(tm.columns):
                ax.text(j, i, f"{tm.loc[row, col]:.2f}", ha="center", va="center", color="w")
        ax.set_title("Transition Matrix")
        fig.tight_layout()

    return tm, {"Occurrences": freqs}


# =============================================================================
# Utils
# =============================================================================


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


# def transition_matrix_plot(tm):
#     """Graph of Transition Matrix

#     Abandonned for now because networkx gives ugly results. Please do help!
#     """
#     try:
#         import networkx as nx
#     except ImportError:
#         raise ImportError(
#             "NeuroKit error: transition_matrix_plot(): the 'networkx' module is required for this ",
#             "function to run. Please install it first (`pip install networkx`).",
#         )

#     # create graph object
#     G = nx.MultiDiGraph(tm)

#     edge_labels = {}
#     for col in tm.columns:
#         for row in tm.index:
#             G.add_edge(row, col, weight=tm.loc[row, col])
#             edge_labels[(row, col)] = label = "{:.02f}".format(tm.loc[row, col])

#     pos = nx.circular_layout(G)
#     nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.5)
#     nx.draw_networkx_edge_labels(G, pos, edge_labels)
#     nx.draw_networkx(G, pos)
