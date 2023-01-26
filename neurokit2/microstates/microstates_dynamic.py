# -*- coding: utf-8 -*-
import pandas as pd

from ..markov import transition_matrix
from ..misc import as_vector


def microstates_dynamic(microstates, show=False):
    """**Dynamic Properties of Microstates**

    This computes statistics related to the transition pattern (based on the
    :func:`.transition_matrix`).

    .. note::

        This function does not compute all the features available under the markov submodule.
        Don't hesitate to open an issue to help us test and decide what features to include.

    Parameters
    ----------
    microstates : np.ndarray
        The topographic maps of the found unique microstates which has a shape of n_channels x
        n_states, generated from :func:`.nk.microstates_segment`.
    show : bool
        Show the transition matrix.


    Returns
    -------
    DataFrame
        Dynamic properties of microstates.

    See Also
    --------
    .transition_matrix, .microstates_static

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      microstates = [0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 0, 0, 2, 2]
      @savefig p_microstates_dynamic1.png scale=100%
      nk.microstates_dynamic(microstates, show=True)
      @suppress
      plt.close()

    """
    # See https://github.com/Frederic-vW/eeg_microstates
    # and https://github.com/maximtrp/mchmm
    # for other implementations

    out = {}

    # Try retrieving info
    if isinstance(microstates, dict):
        microstates = microstates["Sequence"]
    # Sanitize
    microstates = as_vector(microstates)

    # Transition matrix
    tm, info = transition_matrix(microstates, show=show)

    for row in tm.index:
        for col in tm.columns:
            out[str(tm.loc[row].name) + "_to_" + str(tm[col].name)] = tm[col][row]

    df = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("Microstate_")

    # TODO:
    # * Chi-square test statistics of the observed microstates against the expected microstates
    # * Symmetry test statistics of the observed microstates against the expected microstates

    return df
