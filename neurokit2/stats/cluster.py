# -*- coding: utf-8 -*-
import numpy as np
import sklearn.cluster
import functools




def cluster_kmeans(data, n_clusters=2, random_state=None, return_function=False, **kwargs):
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
    >>> # Cluster and visualize
    >>> clustering, clusters = nk.cluster_kmeans(data, n_clusters=3)
    >>> plt.scatter(data[:, 2], data[:, 3], c=clustering.astype(np.float))
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
    clustering_function = functools.partial(cluster_kmeans,
                                            n_clusters=n_clusters,
                                            random_state=random_state,
                                            return_function=False,
                                            **kwargs)


    return clustering, clusters, clustering_function
