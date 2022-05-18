# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg


def fit_loess(y, X=None, alpha=0.75, order=2):
    """**Local Polynomial Regression (LOESS)**

    Performs a LOWESS (LOcally WEighted Scatter-plot Smoother) regression.


    Parameters
    ----------
    y : Union[list, np.array, pd.Series]
        The response variable (the y axis).
    X : Union[list, np.array, pd.Series]
        Explanatory variable (the x axis). If ``None``, will treat y as a continuous signal (useful
        for smoothing).
    alpha : float
        The parameter which controls the degree of smoothing, which corresponds to the proportion
        of the samples to include in local regression.
    order : int
        Degree of the polynomial to fit. Can be 1 or 2 (default).

    Returns
    -------
    array
        Prediction of the LOESS algorithm.
    dict
        Dictionary containing additional information such as the parameters (``order`` and ``alpha``).

    See Also
    ----------
    signal_smooth, signal_detrend, fit_error

    Examples
    ---------
    .. ipython:: python

      import pandas as pd
      import neurokit2 as nk

      # Simulate Signal
      signal = np.cos(np.linspace(start=0, stop=10, num=1000))

      # Add noise to signal
      distorted = nk.signal_distort(signal,
                                    noise_amplitude=[0.3, 0.2, 0.1],
                                    noise_frequency=[5, 10, 50])

      # Smooth signal using local regression
      @savefig p_fit_loess1.png scale=100%
      pd.DataFrame({ "Raw": distorted, "Loess_1": nk.fit_loess(distorted, order=1)[0],
                     "Loess_2": nk.fit_loess(distorted, order=2)[0]}).plot()
      @suppress
      plt.close()

    References
    ----------
    * https://simplyor.netlify.com/loess-from-scratch-in-python-animation.en-us/

    """
    if X is None:
        X = np.linspace(0, 100, len(y))

    assert order in [1, 2], "Deg has to be 1 or 2"
    assert 0 < alpha <= 1, "Alpha has to be between 0 and 1"
    assert len(X) == len(y), "Length of X and y are different"

    X_domain = X

    n = len(X)
    span = int(np.ceil(alpha * n))

    y_predicted = np.zeros(len(X_domain))
    x_space = np.zeros_like(X_domain)

    for i, val in enumerate(X_domain):
        distance = abs(X - val)
        sorted_dist = np.sort(distance)
        ind = np.argsort(distance)

        Nx = X[ind[:span]]
        Ny = y[ind[:span]]

        delx0 = sorted_dist[span - 1]

        u = distance[ind[:span]] / delx0
        w = (1 - u**3) ** 3

        W = np.diag(w)
        A = np.vander(Nx, N=1 + order)

        V = np.matmul(np.matmul(A.T, W), A)
        Y = np.matmul(np.matmul(A.T, W), Ny)
        Q, R = scipy.linalg.qr(V)
        p = scipy.linalg.solve_triangular(R, np.matmul(Q.T, Y))

        y_predicted[i] = np.polyval(p, val)
        x_space[i] = val

    return y_predicted, {"alpha": alpha, "order": order}
