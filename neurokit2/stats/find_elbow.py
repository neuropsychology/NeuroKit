-*- coding: utf-8 -*-
# import numpy as np


# def find_elbow(x, show=False):
#    """
#    Examples
#    ---------
#    >>> import neurokit2 as nk
#    >>> import matplotlib.pyplot as plt
#    >>> import sklearn.datasets
#    >>>
#    >>> # Load the iris dataset
#    >>> data = sklearn.datasets.load_iris().data
#    >>>
#    >>> # How many clusters
#    >>> results = nk.cluster_findnumber(data)
#    >>> x = results["Score_VarianceExplained"]
#    >>>
#    >>>
#    >>> # Plot
#    >>> results.plot(x="n_Clusters", y="Score_VarianceExplained")
#    """
#    # Scale
#    x = (x + np.min(x)) / np.max(x)
#
#    curve = nk.fit_polynomial(x, order=2)
#    diff = np.append(np.diff(x), [np.nan])
#
#    np.where(diff < 0.1)
#
#    gradient = np.gradient(x)
#
#    if show is True:
#        x_range = range(len(x))
#        plt.plot(x_range, x)
#        plt.plot(x_range, diff)
#        plt.plot(x_range, curve)
#        plt.plot(x_range, gradient)
