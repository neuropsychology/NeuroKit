# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .fit_error import fit_rmse


def fit_polynomial(y, X=None, order=2):
    """
    Polynomial Regression.

    Performs a polynomial regression of given order.


    Parameters
    ----------
    y : list, array or Series
        The response variable (the y axis).
    X : list, array or Series
        Explanatory variable (the x axis). If 'None', will treat y as a continuous signal.
    order : int
        The order of the polynomial. 0, 1 or > 1 for a baseline, linear or polynomial
        fit, respectively. Can also be 'auto', it which case it will attempt to find
        the optimal order to minimize the RMSE.

    Returns
    -------
    array
        Prediction of the regression.

    See Also
    ----------
    signal_detrend, fit_error

    Examples
    ---------
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> y = np.cos(np.linspace(start=0, stop=10, num=100))
    >>>
    >>> pd.DataFrame({ "y": y, "Poly_0": nk.fit_polynomial(y, order=0), "Poly_1": nk.fit_polynomial(y, order=1), "Poly_2": nk.fit_polynomial(y, order=2), "Poly_3": nk.fit_polynomial(y, order=3), "Poly_5": nk.fit_polynomial(y, order=5), "Poly_auto": nk.fit_polynomial(y, order='auto')}).plot() #doctest: +SKIP

    """
    if X is None:
        X = np.linspace(0, 100, len(y))

    # Optimal order
    if isinstance(order, str):
        order = fit_polynomial_findorder(y, X, max_order=6)

    # Make prediction
    y_predicted = _fit_polynomial(y, X, order=order)

    return y_predicted


# =============================================================================
# Find order
# =============================================================================
def fit_polynomial_findorder(y, X, max_order=6):
    # TODO: add cross-validation or some kind of penalty to prevent over-fitting?
    best_rmse = 0
    for order in range(max_order):
        y_predicted = _fit_polynomial(y, X, order=order)
        rmse = fit_rmse(y, y_predicted)
        if rmse < best_rmse or best_rmse == 0:
            best_order = order
    return best_order


# =============================================================================
# Internals
# =============================================================================


def _fit_polynomial(y, X, order=2):
    # Generating weights and model for polynomial function with a given degree
    y_predicted = np.polyval(np.polyfit(X, y, order), X)
    return y_predicted
