# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy
import scipy.spatial




def distance(X=None, method="mahalanobis"):
    """Distance

    Compute distance using different metrics.


    Parameters
    ----------
    X : array or DataFrame
        A dataframe of values.
    method : str
        The method to use. One of 'mahalanobis'.

    Returns
    -------
    array
        DataFrame containing the probability of belongning to each cluster.


    Examples
    ---------
    >>> from sklearn import datasets
    >>> import neurokit2 as nk
    >>>
    >>> X = datasets.load_iris().data
    >>> nk.distance(X)
    """
    if isinstance(X, pd.DataFrame) is False:
        X = pd.DataFrame(X)

    method = method.lower()  # remove capitalised letters
    if method in ["mahalanobis"]:
        dist = _distance_mahalanobis(X)
    else:
        raise ValueError("NeuroKit error: distance(): 'method' should be "
                         "one of 'mahalanobis'.")

    return dist



def _distance_mahalanobis(X=None):
    cov = X.cov().values
    cov = scipy.linalg.inv(cov)

    col_means = X.mean().values

    dist = np.full(len(X), np.nan)
    for i in range(X.shape[0]):
        dist[i] = scipy.spatial.distance.mahalanobis(X.iloc[i,:], col_means, cov) ** 2
    return dist