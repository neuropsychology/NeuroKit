import numpy as np
import pandas as pd

from ..signal import signal_autocor


def complexity_relativeroughness(signal, **kwargs):
    """**Relative Roughness (RR)**

    Relative Roughness is a ratio of local variance (autocovariance at lag-1) to global variance
    (autocovariance at lag-0) that can be used to classify different 'noises'
    (see `Hasselman, 2019 <https://complexity-methods.github.io/book/relative-roughness.html>`_).
    It can also be used as an index to test for the applicability of fractal analysis (see
    `Marmelat et al., 2012 <https://doi.org/10.3389/fphys.2012.00208>`_).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    **kwargs : optional
        Other arguments to be passed to ``nk.signal_autocor()``.

    Returns
    --------
    rr : float
        The RR value.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute RR.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      signal = [1, 2, 3, 4, 5]
      rr, _ = nk.complexity_relativeroughness(signal)
      rr

    References
    ----------
    * Marmelat, V., Torre, K., & Delignieres, D. (2012). Relative roughness:
      an index for testing the suitability of the monofractal model.
      Frontiers in Physiology, 3, 208.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    _, acov = signal_autocor(signal, **kwargs)  # Retrieve the dict
    acov = acov["ACov"][0:2]  # Extract cov at lag 0 and 1

    # RR formula
    return 2 * (1 - acov[1] / acov[0]), {"ACov": acov}
