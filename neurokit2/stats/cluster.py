# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.mixture
import scipy.spatial
import functools


def cluster(data, method="kmeans", n_clusters=2, random_state=None, return_function=False, **kwargs):
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
    >>> # Cluster using different methods
    >>> clustering_kmeans, clusters_kmeans = nk.cluster(data, method="kmeans", n_clusters=3)
    >>> clustering_spectral, clusters_spectral = nk.cluster(data, method="spectral", n_clusters=3)
    >>> clustering_hierarchical, clusters_hierarchical = nk.cluster(data, method="hierarchical", n_clusters=3)
    >>> clustering_agglomerative, clusters_agglomerative = nk.cluster(data, method="agglomerative", n_clusters=3)
    >>> clustering_mixture, clusters_mixture = nk.cluster(data, method="mixture", n_clusters=3)
    >>>
    >>> # Visualize classification and 'average cluster'
    >>> fig, axes = plt.subplots(ncols=2, nrows=3)
    >>> axes[0, 0].scatter(data[:, 2], data[:, 3], c=clustering_kmeans['Cluster'])
    >>> axes[0, 0].scatter(clusters_kmeans[:, 2], clusters_kmeans[:, 3], c='red')
    >>> axes[0, 0].set_title("k-means")
    >>> axes[0, 1].scatter(data[:, 2], data[:, 3], c=clustering_spectral['Cluster'])
    >>> axes[0, 1].scatter(clusters_spectral[:, 2], clusters_spectral[:, 3], c='red')
    >>> axes[0, 1].set_title("Spectral")
    >>> axes[1, 0].scatter(data[:, 2], data[:, 3], c=clustering_hierarchical['Cluster'])
    >>> axes[1, 0].scatter(clusters_hierarchical[:, 2], clusters_hierarchical[:, 3], c='red')
    >>> axes[1, 0].set_title("Hierarchical")
    >>> axes[1, 1].scatter(data[:, 2], data[:, 3], c=clustering_agglomerative['Cluster'])
    >>> axes[1, 1].scatter(clusters_agglomerative[:, 2], clusters_agglomerative[:, 3], c='red')
    >>> axes[1, 1].set_title("Agglomerative")
    >>> axes[2, 0].scatter(data[:, 2], data[:, 3], c=clustering_mixture['Cluster'])
    >>> axes[2, 0].scatter(clusters_mixture[:, 2], clusters_mixture[:, 3], c='red')
    >>> axes[2, 0].set_title("Mixture")
    """
    method = method.lower()
    if method in ["kmeans", "k", "k-means", "kmean"]:
        out =  _cluster_kmeans(data,
                               n_clusters=n_clusters,
                               random_state=random_state,
                               return_function=return_function,
                               **kwargs)
    elif method in ["mixture", "mixt"]:
        out =  _cluster_mixture(data,
                               n_clusters=n_clusters,
                               random_state=random_state,
                               return_function=return_function,
                               **kwargs)
    else:
        out =  _cluster_sklearn(data,
                                n_clusters=n_clusters,
                                return_function=return_function,
                                **kwargs)

    return out



# =============================================================================
# Methods
# =============================================================================

def _cluster_kmeans(data, n_clusters=2, random_state=None, return_function=False, **kwargs):
    """K-means
    """
    # Initialize clustering function
    clustering_function = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                                 random_state=random_state,
                                                 **kwargs)

    # Fit
    clustering = clustering_function.fit_predict(data)

    # Get representatives (identical to _cluster_getclusters(), but why recompute when already included)
    clusters = clustering_function.cluster_centers_

    # Get distance
    prediction = _cluster_getdistance(data, clusters)
    prediction["Cluster"] = clustering

    # Return the things
    if return_function is False:
        return prediction, clusters

    # Else, copy function
    clustering_function = functools.partial(_cluster_kmeans,
                                            n_clusters=n_clusters,
                                            random_state=random_state,
                                            return_function=False,
                                            **kwargs)

    return prediction, clusters, clustering_function


def _cluster_sklearn(data, method="spectral", n_clusters=2, return_function=False, **kwargs):
    """Spectral clustering
    """
    # Initialize clustering function
    if method in ["spectral"]:
        clustering_function = sklearn.cluster.SpectralClustering(n_clusters=n_clusters, **kwargs)
    elif method in ["hierarchical", "ward"]:
        clustering_function = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", **kwargs)
    elif method in ["agglomerative", "single"]:
        clustering_function = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage="single", **kwargs)

    # Fit
    clustering = clustering_function.fit_predict(data)

    # Get representatives
    clusters = _cluster_getclusters(data, clustering)

    # Get distance
    prediction = _cluster_getdistance(data, clusters)
    prediction["Cluster"] = clustering

    # Return the things
    if return_function is False:
        return prediction, clusters

    # Else, copy function
    clustering_function = functools.partial(_cluster_sklearn,
                                            n_clusters=n_clusters,
                                            return_function=False,
                                            **kwargs)

    return prediction, clusters, clustering_function


def _cluster_mixture(data, n_clusters=2, random_state=None, return_function=False, **kwargs):
    """Mixture model
    """
    # Initialize clustering function
    clustering_function = sklearn.mixture.GaussianMixture(n_components=n_clusters,
                                                          random_state=random_state,
                                                          **kwargs)
    # Fit
    clustering = clustering_function.fit_predict(data)

    # Get representatives
    clusters = clustering_function.means_

    # Get probability
    prediction = clustering_function.predict_proba(data)
    prediction = pd.DataFrame(prediction).add_prefix("Probability_")
    prediction["Cluster"] = clustering

    # Return the things
    if return_function is False:
        return prediction, clusters

    # Else, copy function
    clustering_function = functools.partial(_cluster_mixture,
                                            n_clusters=n_clusters,
                                            random_state=random_state,
                                            return_function=False,
                                            **kwargs)

    return prediction, clusters, clustering_function

# =============================================================================
# Utils
# =============================================================================
def _cluster_getdistance(data, clusters):
    """Distance between samples and clusters
    """
    distance = scipy.spatial.distance.cdist(data, clusters)
    distance = pd.DataFrame(distance).add_prefix("Distance_")
    return distance


def _cluster_getclusters(data, clustering):
    """Get average representatives of clusters
    """
    n_clusters = len(np.unique(clustering))
    return np.asarray([np.mean(data[np.where(clustering == i)], axis=0) for i in range(n_clusters)])
