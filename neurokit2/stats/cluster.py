# -*- coding: utf-8 -*-
import numpy as np
import sklearn.cluster
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
    >>>
    >>> # Visualize classification and 'average cluster'
    >>> fig, axes = plt.subplots(ncols=2, nrows=2)
    >>> axes[0, 0].scatter(data[:, 2], data[:, 3], c=clustering_kmeans)
    >>> axes[0, 0].scatter(clusters_kmeans[:, 2], clusters_kmeans[:, 3], c='red')
    >>> axes[0, 0].set_title("k-means")
    >>> axes[0, 1].scatter(data[:, 2], data[:, 3], c=clustering_spectral)
    >>> axes[0, 1].scatter(clusters_spectral[:, 2], clusters_spectral[:, 3], c='red')
    >>> axes[0, 1].set_title("Spectral")
    >>> axes[1, 0].scatter(data[:, 2], data[:, 3], c=clustering_hierarchical)
    >>> axes[1, 0].scatter(clusters_hierarchical[:, 2], clusters_hierarchical[:, 3], c='red')
    >>> axes[1, 0].set_title("Hierarchical")
    >>> axes[1, 1].scatter(data[:, 2], data[:, 3], c=clustering_agglomerative)
    >>> axes[1, 1].scatter(clusters_agglomerative[:, 2], clusters_agglomerative[:, 3], c='red')
    >>> axes[1, 1].set_title("Agglomerative")
    """
    method = method.lower()
    if method in ["kmeans", "k", "k-means", "kmean"]:
        out =  _cluster_kmeans(data,
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

    # Get representatives (identical to _cluster_getclusters())
    clusters = clustering_function.cluster_centers_

    # Return the things
    if return_function is False:
        return clustering, clusters

    # Else, copy function
    clustering_function = functools.partial(_cluster_kmeans,
                                            n_clusters=n_clusters,
                                            random_state=random_state,
                                            return_function=False,
                                            **kwargs)


    return clustering, clusters, clustering_function


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

    # Return the things
    if return_function is False:
        return clustering, clusters

    # Else, copy function
    clustering_function = functools.partial(_cluster_sklearn,
                                            n_clusters=n_clusters,
                                            return_function=False,
                                            **kwargs)

    return clustering, clusters, clustering_function




# =============================================================================
# Utils
# =============================================================================

def _cluster_getclusters(data, clustering):
    """Get average representatives of clusters
    """
    n_clusters = len(np.unique(clustering))
    return np.asarray([np.mean(data[np.where(clustering == i)], axis=0) for i in range(n_clusters)])
