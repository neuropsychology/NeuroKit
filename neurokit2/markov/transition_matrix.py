# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import as_vector


def transition_matrix(sequence, order=1, adjust=True, show=False):
    """**Transition Matrix**

    A **Transition Matrix** (also known as a stochastic matrix or a **Markov matrix**) is a
    convenient way of representing and describing a sequence of (discrete) states, also known as
    **discrete Markov chains**. Each of its entries is a probability of transitioning from one
    state to the other.

    .. note::

        This function is fairly new and hasn't be tested extensively. Please help us by
        double-checking the code and letting us know if everything is correct.

    Parameters
    ----------
    sequence : Union[list, np.array, pd.Series]
        A list of discrete states.
    order : int
        The order of the Markov chain.
    adjust : bool
        If ``True``, the transition matrix will be adjusted to ensure that the sum of each row is
        equal to 1. This is useful when the transition matrix is used to represent a probability
        distribution.
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

      sequence = ["A", "A", "C", "B", "B", "B", "C", "A", "A", "D"]

      @savefig p_transition_matrix1.png scale=100%
      tm, _ = nk.transition_matrix(sequence, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      tm

    In this example, the transition from D is unknown (it is the last element), resulting in an
    absence of transitioning probability. As this can cause issues, unknown probabilities are
    replaced by a uniform distribution, but this can be turned off using the ``adjust`` argument.

    .. ipython:: python

      tm, _ = nk.transition_matrix(sequence, adjust=False)
      tm

    Transition matrix of higher order

    .. ipython:: python

      sequence = ["A", "A", "A", "B", "A", "A", "B", "A", "A", "B"]
      tm, _ = nk.transition_matrix(sequence, order=2)
      tm

    """
    sequence = as_vector(sequence)

    # Observed transition matrix
    states = np.unique(sequence)
    n_states = len(states)

    # Get observed transition matrix
    freqs = np.zeros((n_states,) * (order + 1))
    for idx in zip(*[sequence[i:] for i in range(order + 1)]):
        idx = tuple([np.argwhere(states == k)[0][0] for k in idx])
        freqs[idx] += 1
    freqs

    # Find rows containing zeros (unknown transition)
    idx = freqs.sum(axis=-1) == 0

    # Fillit with uniform probability to avoid problem in division
    freqs[idx, :] = 1

    # Convert to probabilities
    tm = (freqs.T / freqs.sum(axis=-1)).T

    # If no adjustment, revert to 0
    freqs[idx, :] = 0
    if adjust is False:
        tm[idx, :] = 0

    # Convert to DataFrame
    if order == 1:
        tm = pd.DataFrame(tm, index=states, columns=states)
        freqs = pd.DataFrame(freqs, index=states, columns=states)

    if show is True:
        if order > 1:
            raise ValueError(
                "Visualization of order > 1 not supported yet. "
                "Consider helping us to implement it!"
            )
        fig, ax = plt.subplots()
        ax.imshow(tm, cmap="Reds", interpolation="nearest")
        ax.set_xticks(np.arange(len(tm)))
        ax.set_yticks(np.arange(len(tm)))
        ax.set_xticklabels(tm.columns)
        ax.set_yticklabels(tm.index)

        # Loop over data dimensions and create text annotations.
        for i, row in enumerate(tm.index):
            for j, col in enumerate(tm.columns):
                if tm.loc[row, col] > 0.5:
                    color = "white"
                else:
                    color = "black"
                ax.text(j, i, f"{tm.loc[row, col]:.2f}", ha="center", va="center", color=color)
        ax.set_title("Transition Matrix")
        fig.tight_layout()

    return tm, {"Occurrences": freqs, "States": states}


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
