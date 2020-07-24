# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.cluster
import sklearn.metrics
import scipy.spatial

from .cluster import cluster
from .cluster_quality import cluster_quality



def cluster_findnumber(data, method="kmeans", n_max=10, show=False, **kwargs):
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
    >>> # How many clusters
    >>> results = nk.cluster_findnumber(data, method="kmeans", show=True)
    """
    results = []
    for i in range(1, n_max):
        # Cluster
        clustering, clusters, info = cluster(data,
                                             method=method,
                                             n_clusters=i,
                                             **kwargs)

        # Compute indices of clustering quality
        _, quality = cluster_quality(data, clustering, clusters, info, **kwargs)
        results.append(quality)

    results = pd.concat(results, axis=0)

    # Gap Diff
    results["Score_GAP_diff"] = results["Score_GAP"] - results["Score_GAP"].shift(-1) + results["Score_GAP_sk"].shift(-1)
    results["Score_GAPmod_diff"] = results["Score_GAPmod"] - results["Score_GAPmod"].shift(-1) + results["Score_GAPmod_sk"].shift(-1)
    results = results.drop(["Score_GAP_sk", "Score_GAPmod_sk"], axis=1)

    if show is True:
        normalized = (results - results.min()) / (results.max() - results.min())
        normalized["n_Clusters"] = np.rint(normalized["n_Clusters"].values * (n_max - 1)) + 1
        normalized.columns = normalized.columns.str.replace('Score', 'Normalized')
        normalized.plot(x="n_Clusters")
    return results
