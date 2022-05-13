# -*- coding: utf-8 -*-
import numpy as np
import sklearn.linear_model
import sklearn.metrics

from .fit_error import fit_rmse


def fit_polynomial(y, X=None, order=2, method="raw"):
    """**Polynomial Regression**

    Performs a polynomial regression of given order.


    Parameters
    ----------
    y : Union[list, np.array, pd.Series]
        The response variable (the y axis).
    X : Union[list, np.array, pd.Series]
        Explanatory variable (the x axis). If ``None``, will treat y as a continuous signal.
    order : int
        The order of the polynomial. 0, 1 or > 1 for a baseline, linear or polynomial fit,
        respectively. Can also be ``"auto"``, in which case it will attempt to find the optimal
        order to minimize the RMSE.
    method : str
        If ``"raw"`` (default), compute standard polynomial coefficients. If ``"orthogonal"``,
        compute orthogonal polynomials (and is equivalent to R's ``poly`` default behavior).

    Returns
    -------
    array
        Prediction of the regression.
    dict
        Dictionary containing additional information such as the parameters (``order``) used and the coefficients (``coefs``).

    See Also
    ----------
    signal_detrend, fit_error, fit_polynomial_findorder

    Examples
    ---------
    .. ipython:: python

      import pandas as pd
      import neurokit2 as nk

      y = np.cos(np.linspace(start=0, stop=10, num=100))
      @savefig p_fit_polynomial1.png scale=100%
      pd.DataFrame({"y": y,
                    "Poly_0": nk.fit_polynomial(y, order=0)[0],
                    "Poly_1": nk.fit_polynomial(y, order=1)[0],
                    "Poly_2": nk.fit_polynomial(y, order=2)[0],
                    "Poly_3": nk.fit_polynomial(y, order=3)[0],
                    "Poly_5": nk.fit_polynomial(y, order=5)[0],
                    "Poly_auto": nk.fit_polynomial(y, order='auto')[0]}).plot()
      @suppress
      plt.close()

    """
    if X is None:
        X = np.linspace(0, 100, len(y))

    # Optimal order
    if isinstance(order, str):
        order = fit_polynomial_findorder(y, X, max_order=6)

    # Make prediction
    if method == "raw":
        y_predicted, coefs = _fit_polynomial(y, X, order=order)
    else:
        y_predicted, coefs = _fit_polynomial_orthogonal(y, X, order=order)

    return y_predicted, {
        "order": order,
        "coefs": coefs,
        "R2": sklearn.metrics.r2_score(y, y_predicted),
    }


# =============================================================================
# Find order
# =============================================================================
def fit_polynomial_findorder(y, X=None, max_order=6):
    """Polynomial Regression.

    Find the optimal order for polynomial fitting. Currently, the only method implemented is
    RMSE minimization.

    Parameters
    ----------
    y : Union[list, np.array, pd.Series]
        The response variable (the y axis).
    X : Union[list, np.array, pd.Series]
        Explanatory variable (the x axis). If 'None', will treat y as a continuous signal.
    max_order : int
        The maximum order to test.

    Returns
    -------
    int
        Optimal order.

    See Also
    ----------
    fit_polynomial

    Examples
    ---------
      import neurokit2 as nk

      y = np.cos(np.linspace(start=0, stop=10, num=100))

      nk.fit_polynomial_findorder(y, max_order=10)
    9

    """
    # TODO: add cross-validation or some kind of penalty to prevent over-fitting?
    if X is None:
        X = np.linspace(0, 100, len(y))

    best_rmse = 0
    for order in range(max_order):
        y_predicted, _ = _fit_polynomial(y, X, order=order)
        rmse = fit_rmse(y, y_predicted)
        if rmse < best_rmse or best_rmse == 0:
            best_order = order
    return best_order


# =============================================================================
# Internals
# =============================================================================


def _fit_polynomial(y, X, order=2):
    coefs = np.polyfit(X, y, order)
    # Generating weights and model for polynomial function with a given degree
    y_predicted = np.polyval(coefs, X)
    return y_predicted, coefs


def _fit_polynomial_orthogonal(y, X, order=2):
    """Fit an orthogonal polynomial regression in Python (equivalent to R's poly())

    from sklearn.datasets import load_iris
    import pandas as pd
    df = load_iris()
    df = pd.DataFrame(data=df.data, columns=df.feature_names)
    y = df.iloc[:, 0].values  # Sepal.Length
    X = df.iloc[:, 1].values  # Sepal.Width
    _fit_polynomial_orthogonal(y, X, order=2)  # doctest: +SKIP
    # Equivalent to R's:
    # coef(lm(Sepal.Length ~ poly(Sepal.Width, 2), data=iris))


    """
    X = np.transpose([X**k for k in range(order + 1)])
    X = np.linalg.qr(X)[0][:, 1:]
    model = sklearn.linear_model.LinearRegression().fit(X, y)
    return model.predict(X), np.insert(model.coef_, 0, model.intercept_)
