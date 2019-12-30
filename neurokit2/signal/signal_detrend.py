# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import scipy.sparse

def signal_detrend(signal, order=1, method="polyonmial", regularization=500):
    """Polynomial detrending of signal.

    Apply a baseline (order = 0), linear (order = 1), or polynomial (order > 1) detrending to the signal (i.e., removing a general trend).

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    order : int
        The order of the polynomial. 0, 1 or > 1 for a baseline ('constant detrend', i.e., remove only the mean), linear (remove the linear trend) or polynomial detrending.
    method : str
        Can be one of 'polynomial' (default; traditional detrending of a given order) or 'tarvainen2002' to use the smoothness priors approach described by Tarvainen (2002) (mostly used in HRV analyses as a lowpass filter to remove complex trends).
    regularization : int
        Only used if `method='tarvainen2002'`. The regularization parameter (default to 500).

    Returns
    -------
    array
        Vector containing the detrended signal.

    See Also
    --------
    signal_filter

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
    >>> pd.DataFrame({"Raw": signal,
                      "Baseline_Detrend": nk.signal_detrend(signal, order=0),
                      "Linear_Detrend": nk.signal_detrend(signal, order=1),
                      "Quadratic_Detrend": nk.signal_detrend(signal, order=2),
                      "Cubic_Detrend": nk.signal_detrend(signal, order=3),
                      "10th_Detrend": nk.signal_detrend(signal, order=10),
                      "Tarvainen_Detrend": nk.signal_detrend(signal, method='tarvainen2002')}).plot()
    >>> plt.axhline(color='k', linestyle='-')

    References
    ----------
    - `Tarvainen, M. P., Ranta-Aho, P. O., & Karjalainen, P. A. (2002). An advanced detrending method with application to HRV analysis. IEEE Transactions on Biomedical Engineering, 49(2), 172-175. <https://ieeexplore.ieee.org/document/979357>`_
    """
    if method.lower() in ["tarvainen", "tarvainen2002"]:
        detrended = _signal_detrend_tarvainen2002(signal, regularization)
    else:
        detrended = _signal_detrend_polynomial(signal, order)

    return detrended






def _signal_detrend_polynomial(signal, order=1):
    x_axis = np.linspace(0, 100, num=len(signal))

    # Generating weights and model for polynomial function with a given degree
    trend = np.polyval(np.polyfit(x_axis, signal, order), x_axis)

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
