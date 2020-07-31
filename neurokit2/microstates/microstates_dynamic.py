# -*- coding: utf-8 -*-
import pandas as pd
from ..complexity import transition_matrix
from ..misc import as_vector


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
    >>> import neurokit2 as nk
    >>> import numpy as np
    >>>
    >>> microstates = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 0, 0])
    >>> nk.microstates_dynamic(microstates) #doctest: +ELIIPSIS
           Microstate_0_to_0  ...  Microstate_Symmetry_p
        0               0.75  ...                    1.0

    [1 rows x 15 columns]

    """
    microstates = as_vector(microstates)
    out = {}

    # Transition matrix
    results = transition_matrix(microstates)
    T = results["Observed"]

    for row in T.index:
        for col in T.columns:
            out[str(T.loc[row].name) + "_to_" + str(T[col].name)] = T[col][row]

    for _, rez in enumerate(results):
        if rez not in ["Observed", "Expected"]:
            out[rez] = results[rez]

    df = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("Microstate_")

    return df
