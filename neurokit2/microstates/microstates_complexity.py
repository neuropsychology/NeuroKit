# -*- coding: utf-8 -*-
import pandas as pd

from ..complexity import entropy_shannon
from ..misc import as_vector


def microstates_complexity(microstates, show=False):
    """**Complexity of Microstates Pattern**

    This computes the complexity related to the sequence of the microstates pattern.

    .. note::

        This function does not compute all the features available under the complexity
        submodule. Don't hesitate to open an issue to help us test and decide what features to
        include.

    Parameters
    ----------
    microstates : np.ndarray
        The topographic maps of the found unique microstates which has a shape of n_channels x
        n_states, generated from :func:`.nk.microstates_segment`.
    show : bool
        Show the transition matrix.

    See Also
    --------
    .microstates_dynamic, .microstates_static

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      microstates = [0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 0, 0, 2, 2]
      @savefig p_microstates_complexity1.png scale=100%
      nk.microstates_complexity(microstates, show=True)
      @suppress
      plt.close()

    """
    # Try retrieving info
    if isinstance(microstates, dict):
        microstates = microstates["Sequence"]
    # Sanitize
    microstates = as_vector(microstates)

    # Initialize output container
    out = {}

    # Empirical Shannon entropy
    out["Entropy_Shannon"], _ = entropy_shannon(microstates, show=show)

    # Maximym entropy given the number of different states
    #    h_max = np.log2(len(np.unique(microstates)))

    df = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("Microstates_")
    return df
