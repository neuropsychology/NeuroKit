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
    >>>
    >>> # Visualize classification and 'average cluster'
    >>> fig, axes = plt.subplots(ncols=2)
    >>> axes[0].scatter(data[:, 2], data[:, 3], c=clustering_kmeans)
    >>> axes[0].scatter(clusters_kmeans[:, 2], clusters_kmeans[:, 3], c='red')
    >>> axes[1].scatter(data[:, 2], data[:, 3], c=clustering_spectral)
    >>> axes[1].scatter(clusters_spectral[:, 2], clusters_spectral[:, 3], c='red')
    """
    if method in ["kmeans", "k", "k-means", "kmean"]:
        out =  _cluster_kmeans(data,
                               n_clusters=n_clusters,
                               random_state=random_state,
                               return_function=return_function,
                               **kwargs)
    elif method in ["spectral"]:
        out =  _cluster_spectral(data,
                                 n_clusters=n_clusters,
                                 return_function=return_function,
                                 **kwargs)

    return out





def _cluster_kmeans(data, n_clusters=2, random_state=None, return_function=False, **kwargs):
    """K-means
    """
    # Initialize clustering function
    clustering_function = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                                 random_state=random_state,
                                                 **kwargs)

    # Fit
    clustering = clustering_function.fit_predict(data)
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


def _cluster_spectral(data, n_clusters=2, return_function=False, **kwargs):
    """Spectral clustering
    """
    # Initialize clustering function
    clustering_function = sklearn.cluster.SpectralClustering(n_clusters=n_clusters, **kwargs)

    # Fit
    clustering = clustering_function.fit_predict(data)
    clusters = np.asarray([np.mean(data[np.where(clustering == i)], axis=0) for i in range(n_clusters)])

    # Return the things
    if return_function is False:
        return clustering, clusters

    # Else, copy function
    clustering_function = functools.partial(_cluster_spectral,
                                            n_clusters=n_clusters,
                                            return_function=False,
                                            **kwargs)



    return clustering, clusters, clustering_function
