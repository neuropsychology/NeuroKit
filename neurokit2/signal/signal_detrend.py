# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.sparse

from ..stats import fit_loess, fit_polynomial


def signal_detrend(signal, method="polynomial", order=1, regularization=500, alpha=0.75, window=1.5, stepsize=0.02):
    """
    Polynomial detrending of signal.

    Apply a baseline (order = 0), linear (order = 1), or polynomial (order > 1) detrending to the signal (i.e., removing a general trend). One can also use other methods, such as smoothness priors approach described by Tarvainen (2002) or LOESS regression, but these scale badly for long signals.

    Parameters
    ----------
    signal : list, array or Series
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        Can be one of 'polynomial' (default; traditional detrending of a given order) or 'tarvainen2002' to use the smoothness priors approach described by Tarvainen (2002) (mostly used in HRV analyses as a lowpass filter to remove complex trends), 'loess' for LOESS smoothing trend removal or 'locreg' for local linear regression (the 'runline' algorithm from chronux).
    order : int
        Only used if `method` is 'polynomial'. The order of the polynomial. 0, 1 or > 1 for a baseline ('constant detrend', i.e., remove only the mean), linear (remove the linear trend) or polynomial detrending, respectively. Can also be 'auto', it which case it will attempt to find the optimal order to minimize the RMSE.
    regularization : int
        Only used if `method='tarvainen2002'`. The regularization parameter (default to 500).
    alpha : float
        Only used if `method` is 'loess'. The parameter which controls the degree of smoothing.
    window, stepsize : float
        Only used if `method` is 'locreg'. The detrending 'window' should correspond to the desired low frequency band to remove multiplied by the sampling rate (for instance, ``1.5*1000`` will remove frequencies below 1.5Hz for a signal sampled at 1000Hz). The 'stepsize' should also be multiplied by the sampling rate.

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
    >>> # Simulate signal with low and high frequency
    >>> signal = nk.signal_simulate(frequency=[0.1, 2], amplitude=[2, 0.5], sampling_rate=100)
    >>> signal = signal + (3 + np.linspace(0, 6, num=len(signal)))  # Add baseline and linear trend
    >>>
    >>> # Apply detrending algorithms
    >>> baseline = nk.signal_detrend(signal, order=0)  # Constant detrend (removes the mean)
    >>> linear = nk.signal_detrend(signal, order=1)  # Linear detrend
    >>> quadratic = nk.signal_detrend(signal, order=2)  # Quadratic detrend
    >>> cubic = nk.signal_detrend(signal, order=3)  # Cubic detrend
    >>> poly10 = nk.signal_detrend(signal, order=10)  # Linear detrend (10th order)
    >>> tarvainen = nk.signal_detrend(signal, method='tarvainen2002')  # Tarvainen (2002) method
    >>> loess = nk.signal_detrend(signal, method='loess')  # LOESS detrend (smooth removal)
    >>> locreg = nk.signal_detrend(signal, method='locreg', window=1.5*100, stepsize=0.02*100)  # Local regression (100Hz)
    >>>
    >>> # Visualize different methods
    >>> axes = pd.DataFrame({"Original signal": signal,
    ...                      "Baseline": baseline,
    ...                      "Linear": linear,
    ...                      "Quadratic": quadratic,
    ...                      "Cubic": cubic,
    ...                      "Polynomial (10th)": poly10,
    ...                      "Tarvainen": tarvainen,
    ...                      "LOESS": loess,
    ...                      "Local Regression": locreg}).plot(subplots=True)
    >>> # Plot horizontal lines to better visualize the detrending
    >>> for subplot in axes: #doctest: +SKIP
    ...     subplot.axhline(y=0, color='k', linestyle='--') #doctest: +SKIP

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
    elif method in ["locdetrend", "runline", "locreg", "locregression"]:
        detrended = _signal_detrend_locreg(signal, window=window, stepsize=stepsize)
    else:
        raise ValueError(
            "NeuroKit error: signal_detrend(): 'method' should be one of 'polynomial', 'loess' or 'tarvainen2002'."
        )

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
    """
    Method by Tarvainen et al., 2002.

    - Tarvainen, M. P., Ranta-Aho, P. O., & Karjalainen, P. A. (2002). An advanced detrending method with application to HRV
    analysis. IEEE Transactions on Biomedical Engineering, 49(2), 172-175.

    """
    N = len(signal)
    identity = np.eye(N)
    B = np.dot(np.ones((N - 2, 1)), np.array([[1, -2, 1]]))
    D_2 = scipy.sparse.dia_matrix((B.T, [0, 1, 2]), shape=(N - 2, N))
    inv = np.linalg.inv(identity + regularization ** 2 * D_2.T @ D_2)
    z_stat = ((identity - inv)) @ signal

    trend = np.squeeze(np.asarray(signal - z_stat))

    # detrend
    detrended = np.array(signal) - trend
    return detrended


def _signal_detrend_locreg(signal, window=1.5, stepsize=0.02):
    """
    Local linear regression ('runline' algorithm from chronux). Based on https://github.com/sappelhoff/pyprep.

    - http://chronux.org/chronuxFiles/Documentation/chronux/spectral_analysis/continuous/locdetrend.html
    - https://github.com/sappelhoff/pyprep/blob/master/pyprep/removeTrend.py
    - https://github.com/VisLab/EEG-Clean-Tools/blob/master/PrepPipeline/utilities/localDetrend.m

    """
    length = len(signal)

    # Sanity chekcs
    window = int(window)
    stepsize = int(stepsize)
    if window > length:
        raise ValueError(
            "NeuroKit error: signal_detrend(): 'window' should be "
            "less than the number of samples. Try using 1.5 * sampling rate."
        )
    if stepsize <= 1:
        raise ValueError("NeuroKit error: signal_detrend(): 'stepsize' should be more than 1. Increase its value.")

    y_line = np.zeros((length, 1))
    norm = np.zeros((length, 1))
    nwin = np.int(np.ceil((length - window) / stepsize))
    yfit = np.zeros((nwin, window))
    xwt = (np.arange(1, window + 1) - window / 2) / (window / 2)
    wt = np.power(1 - np.power(np.absolute(xwt), 3), 3)
    for j in range(0, nwin):
        tseg = signal[(stepsize * j) : (stepsize * j + window)]
        y1 = np.mean(tseg)
        y2 = np.mean(np.multiply(np.arange(1, window + 1), tseg)) * (2 / (window + 1))
        a = np.multiply(np.subtract(y2, y1), 6 / (window - 1))
        b = np.subtract(y1, a * (window + 1) / 2)
        yfit[j, :] = np.multiply(np.arange(1, window + 1), a) + b
        y_line[(j * stepsize) : (j * stepsize + window)] = y_line[
            (j * stepsize) : (j * stepsize + window)
        ] + np.reshape(np.multiply(yfit[j, :], wt), (window, 1))
        norm[(j * stepsize) : (j * stepsize + window)] = norm[(j * stepsize) : (j * stepsize + window)] + np.reshape(
            wt, (window, 1)
        )

    above_norm = np.where(norm[:, 0] > 0)
    y_line[above_norm] = y_line[above_norm] / norm[above_norm]

    indx = (nwin - 1) * stepsize + window - 1
    npts = length - indx + 1
    y_line[indx - 1 :] = np.reshape((np.multiply(np.arange(window + 1, window + npts + 1), a) + b), (npts, 1))

    detrended = signal - y_line[:, 0]
    return detrended
