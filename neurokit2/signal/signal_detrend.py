# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import scipy.sparse

from ..stats import fit_loess
from ..stats import fit_polynomial


def signal_detrend(signal, method="polynomial", order=1, regularization=500, alpha=0.75):
    """Polynomial detrending of signal.

    Apply a baseline (order = 0), linear (order = 1), or polynomial (order > 1) detrending to the signal (i.e., removing a general trend). One can also use other methods, such as smoothness priors approach described by Tarvainen (2002) or LOESS regression, but these scale badly for long signals.

    Parameters
    ----------
    signal : list, array or Series
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        Can be one of 'polynomial' (default; traditional detrending of a given order) or 'tarvainen2002' to use the smoothness priors approach described by Tarvainen (2002) (mostly used in HRV analyses as a lowpass filter to remove complex trends), or 'loess' for LOESS smoothing trend removal.
    order : int
        Only used if `method` is 'polynomial'. The order of the polynomial. 0, 1 or > 1 for a baseline ('constant detrend', i.e., remove only the mean), linear (remove the linear trend) or polynomial detrending, respectively. Can also be 'auto', it which case it will attempt to find the optimal order to minimize the RMSE.
    regularization : int
        Only used if `method='tarvainen2002'`. The regularization parameter (default to 500).
    alpha : float
        Only used if `method` is 'loess'. The parameter which controls the degree of smoothing.

    Returns
    -------
    array
        Vector containing the detrended signal.

    See Also
    --------
    signal_filter, fit_loess

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>> import matplotlib.pyplot as plt
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=10, num=1000))  # Low freq
    >>> signal += np.cos(np.linspace(start=0, stop=100, num=1000))  # High freq
    >>> signal += 3  # Add baseline
    >>>
    >>> axes = pd.DataFrame({"Raw": signal,
                      "Baseline": nk.signal_detrend(signal, order=0),
                      "Linear": nk.signal_detrend(signal, order=1),
                      "Quadratic": nk.signal_detrend(signal, order=2),
                      "Cubic": nk.signal_detrend(signal, order=3),
                      "10th": nk.signal_detrend(signal, order=10),
                      "Tarvainen": nk.signal_detrend(signal, method='tarvainen2002'),
                      "LOESS": nk.signal_detrend(signal, method='loess')}).plot(subplots=True)
    >>> # Plot horizontal lines to better visualize the detrending
    >>> for subplot in axes:
    >>>     subplot.axhline(y=0, color='k', linestyle='--')

    References
    ----------
    - `Tarvainen, M. P., Ranta-Aho, P. O., & Karjalainen, P. A. (2002). An advanced detrending method with application to HRV analysis. IEEE Transactions on Biomedical Engineering, 49(2), 172-175. <https://ieeexplore.ieee.org/document/979357>`_
    """
    method = method.lower()
    if method in ["tarvainen", "tarvainen2002"]:
        detrended = _signal_detrend_tarvainen2002(signal, regularization)
    elif method in ["poly", "polynomial"]:
        detrended = _signal_detrend_polynomial(signal, order)
    elif method in ["loess", "lowess"]:
        detrended = _signal_detrend_loess(signal, alpha=alpha)
    else:
        raise ValueError("NeuroKit error: signal_detrend(): 'method' should be "
                         "one of 'polynomial', 'loess' or 'tarvainen2002'.")

    return detrended




# =============================================================================
# Internals
# =============================================================================
def _signal_detrend_loess(signal, alpha=0.75):
    detrended = np.array(signal) - fit_loess(signal, alpha=alpha)
    return detrended





def _signal_detrend_polynomial(signal, order=1):
    # Get polynomial fit
    trend = fit_polynomial(signal, X=None, order=order)

    # detrend
    detrended = np.array(signal) - trend
    return detrended




def _signal_detrend_tarvainen2002(signal, regularization=500):
    """`Tarvainen, M. P., Ranta-Aho, P. O., & Karjalainen, P. A. (2002). An advanced detrending method with application to HRV analysis. IEEE Transactions on Biomedical Engineering, 49(2), 172-175. <https://ieeexplore.ieee.org/document/979357>`_
    """
    N = len(signal)
    identity = np.eye(N)
    B = np.dot(np.ones((N-2, 1)), np.array([[1, -2, 1]]))
    D_2 = scipy.sparse.dia_matrix((B.T, [0, 1, 2]), shape=(N-2, N))
    inv = np.linalg.inv(identity + regularization**2 * D_2.T @ D_2)
    z_stat = ((identity - inv)) @ signal

    trend = np.squeeze(np.asarray(signal - z_stat))

    # detrend
    detrended = np.array(signal) - trend
    return detrended
