# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sklearn.mixture


def fit_mixture(X=None, n_clusters=2):
    """
    Gaussian Mixture Model.

    Performs a polynomial regression of given order.


    Parameters
    ----------
    X : list, array or Series
        The values to classify.
    n_clusters : int
        Number of components to look for.
    threshold : float
        Probability threshold to

    Returns
    -------
    pd.DataFrame
        DataFrame containing the probability of belongning to each cluster.

    See Also
    ----------
    signal_detrend, fit_error

    Examples
    ---------
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> x = nk.signal_simulate()
    >>> probs = nk.fit_mixture(x, n_clusters=2)
    >>> fig = nk.signal_plot([x, probs["Cluster_0"], probs["Cluster_1"]], standardize=True)
    >>> fig #doctest: +SKIP

    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # fit a Gaussian Mixture Model with two components
    clf = sklearn.mixture.GaussianMixture(n_components=n_clusters, random_state=333)
    clf = clf.fit(X)

    # Get predicted probabilities
    predicted = clf.predict_proba(X)
    probabilities = pd.DataFrame(predicted).add_prefix("Cluster_")

    return probabilities
