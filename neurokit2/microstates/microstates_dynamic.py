# -*- coding: utf-8 -*-
import pandas as pd
from ..complexity import transition_matrix
from ..misc import as_vector


def microstates_dynamic(microstates):
    """Dynamic properties of microstates (transition pattern)

    Based on https://github.com/Frederic-vW/eeg_microstates and https://github.com/maximtrp/mchmm

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> microstates = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 0, 0])
    >>> nk.microstates_dynamic(microstates)
    """
    microstates = as_vector(microstates)
    out = {}

    # Transition matrix
    results = transition_matrix(microstates)
    T = results["Observed"]

    for row in T.index:
        for col in T.columns:
            out[str(T.loc[row].name) + "_to_" + str(T[col].name)] = T[col][row]

    for rez in results.keys():
        if rez not in ["Observed", "Expected"]:
            out[rez] = results[rez]

    df = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("Microstate_")

    return df





