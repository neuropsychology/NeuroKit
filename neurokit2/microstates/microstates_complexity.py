# -*- coding: utf-8 -*-
import pandas as pd
from ..misc import as_vector
from ..complexity import entropy_shannon


def microstates_complexity(microstates):
    """Complexity of microstates pattern

    """
    microstates = as_vector(microstates)
    out = {}

    # Empirical Shannon entropy
    out["Entropy_Shannon"] = entropy_shannon(microstates)

    # Maximym entropy given the number of different states
#    h_max = np.log2(len(np.unique(microstates)))

    df = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("Microstate_")
    return df
