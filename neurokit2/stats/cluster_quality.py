# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.cluster
import sklearn.metrics
import sklearn.mixture
import scipy.spatial


def cluster_quality(data, clustering, clusters=None, info=None, n_random=10):
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
    >>> clustering, clusters, info = nk.cluster(data, method="kmeans", n_clusters=3)
    >>>
    >>> # Compute indices of clustering quality
    >>> individual, general = nk.cluster_quality(data, clustering, clusters, info)
    >>> general
    """
    if isinstance(clustering, tuple):
        clustering, clusters, info = clustering

    n_clusters = len(clusters)
    clustering = clustering["Cluster"]

    # Individual
    individual = {}
    if n_clusters == 1:
        individual["Clustering_Silhouette"] = np.full(len(clustering), np.nan)
    else:
        individual["Clustering_Silhouette"] = sklearn.metrics.silhouette_samples(data, clustering)

    distance = _cluster_quality_distance(data, clusters)
    distance = {"Clustering_Distance_" + str(i): distance[:, i] for i in range(distance.shape[1])}
    individual.update(distance)
    individual = pd.DataFrame(individual)

    # General
    general = {"n_Clusters": n_clusters}
    if n_clusters == 1:
        general["Score_Silhouette"] = np.nan
        general["Score_Calinski"] = np.nan
        general["Score_Bouldin"] = np.nan
    else:
        general["Score_Silhouette"] = sklearn.metrics.silhouette_score(data, clustering)
        general["Score_Calinski"] = sklearn.metrics.calinski_harabasz_score(data, clustering)
        general["Score_Bouldin"] = sklearn.metrics.davies_bouldin_score(data, clustering)

    # Variance explained
    general["Score_VarianceExplained"] = _cluster_quality_variance(data, clustering, clusters)

    # Gap statistic
    general["Score_GAP"] = _cluster_quality_gap(data,
                                                clusters,
                                                info["clustering_function"],
                                                n_random=n_random)

    # Mixture models
    if "sklearn_model" in info:
        if isinstance(info["sklearn_model"], sklearn.mixture.GaussianMixture):
            general["Score_AIC"] = info["sklearn_model"].aic(data)
            general["Score_BIC"] = info["sklearn_model"].bic(data)
            general["Score_LogLikelihood"] = info["sklearn_model"].score(data)

    general = pd.DataFrame.from_dict(general, orient="index").T
    return individual, general



# =============================================================================
# Utils
# =============================================================================
def _cluster_quality_distance(data, clusters):
    """Distance between samples and clusters
    """
    distance = scipy.spatial.distance.cdist(data, clusters)
    return distance



def _cluster_quality_sumsquares(data, clusters):
    """Within-clusters sum of squares
    """
    min_distance = np.min(_cluster_quality_distance(data, clusters), axis=1)
    return np.sum(min_distance**2)




def _cluster_quality_variance(data, clustering, clusters):
    """Variance explained by clustering
    """
    sum_squares_within =_cluster_quality_sumsquares(data, clusters)
    sum_squares_total = np.sum(scipy.spatial.distance.pdist(data)**2)/data.shape[0]
    return (sum_squares_total - sum_squares_within) / sum_squares_total




def _cluster_quality_gap(data, clusters, clustering_function, n_random=10):
    """GAP statistic
    """
    sum_squares_within = _cluster_quality_sumsquares(data, clusters)

    random_data = scipy.random.random_sample(size=(*data.shape, n_random))
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    sum_squares_random = np.zeros(random_data.shape[-1])
    for i in range(len(sum_squares_random)):
        random_data[..., i] = random_data[..., i] * scipy.matrix(np.diag(maxs - mins)) + mins
        random_clustering, random_clusters, info = clustering_function(random_data[..., i])
        sum_squares_random[i] = _cluster_quality_sumsquares(random_data[..., i], random_clusters)
    gap = np.log(np.mean(sum_squares_random))-np.log(sum_squares_within)
    return gap
