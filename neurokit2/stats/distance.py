# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy
import scipy.spatial

from .standardize import standardize


def distance(X=None, method="mahalanobis"):
    """**Distance**

    Compute distance using different metrics.

    Parameters
    ----------
    X : array or DataFrame
        A dataframe of values.
    method : str
        The method to use. One of ``"mahalanobis"`` or ``"mean"`` for the average distance from the
        mean.

    Returns
    -------
    array
        Vector containing the distance values.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      # Load the iris dataset
      data = nk.data("iris").drop("Species", axis=1)
      data["Distance"] = nk.distance(data, method="mahalanobis")

      @savefig p_distance1.png scale=100%
      fig = data.plot(x="Petal.Length", y="Petal.Width", s="Distance", c="Distance", kind="scatter")
      @suppress
      plt.close()

    .. ipython:: python

      data["DistanceZ"] = np.abs(nk.distance(data.drop("Distance", axis=1), method="mean"))

      @savefig p_distance2.png scale=100%
      fig = data.plot(x="Petal.Length", y="Sepal.Length", s="DistanceZ", c="DistanceZ", kind="scatter")
      @suppress
      plt.close()

    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    method = method.lower()  # remove capitalised letters
    if method in ["mahalanobis"]:
        dist = _distance_mahalanobis(X)
    elif method in ["mean", "center", "average"]:
        dist = _distance_mean(X)
    else:
        raise ValueError("NeuroKit error: distance(): 'method' should be one of 'mahalanobis'.")

    return dist


# =============================================================================
# Methods
# =============================================================================


def _distance_mahalanobis(X=None):
    cov = X.cov().values
    cov = scipy.linalg.inv(cov)

    col_means = X.mean().values

    dist = np.full(len(X), np.nan)
    for i in range(len(X)):
        dist[i] = scipy.spatial.distance.mahalanobis(X.iloc[i, :].values, col_means, cov) ** 2
    return dist


def _distance_mean(X=None):
    Z = standardize(X)
    dist = Z.mean(axis=1).values
    return dist
