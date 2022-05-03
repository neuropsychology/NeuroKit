# -*- coding: utf-8 -*-
import pandas as pd

from ..markov import transition_matrix


def microstates_dynamic(microstates):
    """Dynamic properties of microstates (transition pattern)

    Based on https://github.com/Frederic-vW/eeg_microstates and https://github.com/maximtrp/mchmm

    Parameters
    ----------
    microstates : np.ndarray
        The topographic maps of the found unique microstates which has a shape of n_channels x n_states,
        generated from ``nk.microstates_segment()``.

    Returns
    -------
    DataFrame
        Dynamic properties of microstates:
        - Results of the observed transition matrix
        - Chi-square test statistics of the observed microstates against the expected microstates
        - Symmetry test statistics of the observed microstates against the expected microstates

    See Also
    --------
    transition_matrix

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      microstates = [0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 0, 0]
      nk.microstates_dynamic(microstates)

    """
    out = {}

    # Transition matrix
    tm, info = transition_matrix(microstates)

    for row in tm.index:
        for col in tm.columns:
            out[str(tm.loc[row].name) + "_to_" + str(tm[col].name)] = tm[col][row]

    # out.update(results)
    # out.pop("Expected")

    df = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("Microstate_")

    return df
