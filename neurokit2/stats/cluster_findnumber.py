# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .cluster import cluster
from .cluster_quality import cluster_quality


def cluster_findnumber(data, method="kmeans", n_max=10, show=False, **kwargs):
    """Find the optimal number of clusters based on different metrices of quality.

    Parameters
    ----------
    data : np.ndarray
        An array (channels, times) of M/EEG data.
    method : str
        The clustering algorithm to be passed into ``nk.cluster()``.
    n_max : int
        Runs the clustering alogrithm from 1 to n_max desired clusters in ``nk.cluster()`` with quality
        metrices produced for each cluster number.
    show : bool
        Plot indices normalized on the same scale.
    **kwargs
        Other arguments to be passed into ``nk.cluster()`` and ``nk.cluster_quality()``.

    Returns
    -------
    DataFrame
        The different quality scores for each number of clusters:
        - Score_Silhouette
        - Score_Calinski
        - Score_Bouldin
        - Score_VarianceExplained
        - Score_GAP
        - Score_GAPmod
        - Score_GAP_diff
        - Score_GAPmod_diff

    See Also
    --------
    cluster, cluster_quality

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Load the iris dataset
    >>> data = nk.data("iris")
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

    results = pd.concat(results, axis=0).reset_index(drop=True)

    # Gap Diff
    results["Score_GAP_diff"] = results["Score_GAP"] - results["Score_GAP"].shift(-1) + results["Score_GAP_sk"].shift(-1)
    results["Score_GAPmod_diff"] = results["Score_GAPmod"] - results["Score_GAPmod"].shift(-1) + results["Score_GAPmod_sk"].shift(-1)
    results = results.drop(["Score_GAP_sk", "Score_GAPmod_sk"], axis=1)

    if show is True:
        normalized = (results - results.min()) / (results.max() - results.min())
        normalized["n_Clusters"] = np.rint(np.arange(1, n_max))
        normalized.columns = normalized.columns.str.replace('Score', 'Normalized')
        normalized.plot(x="n_Clusters")
    return results
