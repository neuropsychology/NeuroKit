import numpy as np
import pandas as pd

from ..signal import signal_autocor


def complexity_rr(signal):
    """Relative Roughness (RR)

    Relative Roughness is a ratio of local variance (autocovariance at lag-1) to global variance
    (autocovariance at lag-0) that can be used to classify different 'noises'
    (see `Hasselman, 2019 <https://complexity-methods.github.io/book/relative-roughness.html>`_).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    References
    ----------
    - Marmelat, V., Torre, K., & Delignieres, D. (2012). Relative roughness: an index for testing the
    suitability of the monofractal model. Frontiers in Physiology, 3, 208.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> signal = [1, 2, 3, 4, 5]
    >>> rr, _ = nk.complexity_rr(signal)
    >>> rr

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    _, acov = signal_autocor(signal)  # Retrieve the dict
    acov = acov["ACov"][0:2]  # Extract cov at lag 0 and 1

    # RR formula
    return 2 * (1 - acov[1] / acov[0]), {"ACov": acov}
