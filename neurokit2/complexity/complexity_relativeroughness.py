from warnings import warn

import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning


def complexity_relativeroughness(signal, demean=True, unbiased=False):
    """**Relative Roughness (RR)**

    Relative Roughness is a ratio of local variance (autocovariance at lag-1) to global variance
    (autocovariance at lag-0) that can be used to classify different 'noises'
    (see `Hasselman, 2019 <https://complexity-methods.github.io/book/relative-roughness.html>`_).
    It can also be used as an index to test for the applicability of fractal analysis (see
    `Marmelat et al., 2012 <https://doi.org/10.3389/fphys.2012.00208>`_).

    .. math::

      RR = 2 \\left(1 - \\frac{\\gamma_1}{\\gamma_0}\\right)

    where :math:`\\gamma_0` and :math:`\\gamma_1` are the autocovariances of the signal at lag 0
    (its variance) and 1. RR is therefore 2 minus twice the lag-1 autocorrelation: it is 2 for white
    noise, :math:`4 - 2^{2H}` for a fractional Gaussian noise of Hurst exponent *H* (from 3 for
    *H* close to 0 to 0 for *H* close to 1), and close to 0 for a fractional Brownian motion, whose
    successive values are strongly correlated.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    demean : bool
        If ``True`` (default), the mean of the signal is subtracted before computing the
        autocovariances, as in :func:`.signal_autocor`.
    unbiased : bool
        If ``True``, the autocovariances are divided by the number of overlapping samples (*N* and
        *N - 1*) instead of being both divided by *N* (the default, as in R's ``acf()``), as in
        :func:`.signal_autocor`.

    Returns
    --------
    rr : float
        The RR value.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute RR (the autocovariances at lag 0 and 1, ``"ACov"``, not divided by *N* unless
        ``unbiased=True``).

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
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    signal = np.asarray(signal, dtype=float)
    n = len(signal)

    if np.any(np.isnan(signal)):
        warn("The signal contains missing values. Returning NaN.", category=NeuroKitWarning)
        return np.nan, {"ACov": np.full(2, np.nan)}
    # Constant signals (checked before demeaning, which can leave rounding errors)
    if n < 2 or (demean and np.all(signal == signal[0])) or not np.any(signal):
        warn("The signal is constant or too short, RR is undefined. Returning NaN.", category=NeuroKitWarning)
        return np.nan, {"ACov": np.full(2, np.nan)}

    if demean:
        signal = signal - np.mean(signal)

    # Autocovariances at lag 0 and 1 (only these two are needed, no need for the full ACF)
    acov = np.array([np.dot(signal, signal), np.dot(signal[:-1], signal[1:])])
    if unbiased:
        acov = acov / np.array([n, n - 1])

    # RR formula
    return 2 * (1 - acov[1] / acov[0]), {"ACov": acov}
