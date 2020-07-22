# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.cluster
import sklearn.metrics
import scipy.spatial

from .cluster import cluster
from .cluster_quality import cluster_quality



def cluster_findnumber(data, n_max=10, show=False):
    """
    Examples
    ----------
    >>> import neurokit2 as nk
    >>> import matplotlib.pyplot as plt
    >>> import sklearn.datasets
    >>>
    >>> # Load the iris dataset
    >>> data = sklearn.datasets.load_iris().data
    >>>
    >>> # Cluster
    >>> nk.cluster_findnumber(data, show=True)
    """
    results = []
    for i in range(1, n_max):
        # Cluster
        clustering, clusters, clustering_function = cluster(data,
                                                            method="kmeans",
                                                            n_clusters=i,
                                                            return_function=True)

        # Compute indices of clustering quality
        _, quality = cluster_quality(data, clustering, clusters, clustering_function)
        results.append(quality)

    results = pd.concat(results, axis=0)

    if show is True:
        normalized = (results - results.min()) / (results.max() - results.min())
        normalized["n_Clusters"] = np.rint(normalized["n_Clusters"].values * (n_max - 1)) + 1
        normalized.plot(x="n_Clusters")
    return results


