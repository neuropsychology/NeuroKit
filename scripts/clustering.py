
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.cluster
import sklearn.metrics
import scipy.spatial
import functools

# import some data to play with
data = sklearn.datasets.load_iris().data
n_clusters=3


# It is possible for a function to return itself with a particular set of parameters
#def myfunc(x=3, y="y"):
#    return x, y, functools.partial(myfunc, x=2, y=y)
#a, b, f = myfunc(x=4, y="z")
#c, d, f = f()



def cluster_kmeans(data, n_clusters=2, random_state=None, **kwargs):
    """
    Examples
    ----------
    >>> import neurokit2 as nk
    >>> import sklearn.datasets
    >>>
    """

    # Initialize clustering function
    clustering_function = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                                 random_state=random_state,
                                                 **kwargs)

    # Fit
    clustering = clustering_function.fit_predict(data)
    clusters = clustering_function.cluster_centers_

    # Copy function
    clustering_function = functools.partial(cluster_kmeans,
                                            data=data,
                                            n_clusters=n_clusters,
                                            random_state=random_state,
                                            **kwargs)


    return clustering, clusters, clustering_function





def cluster_quality(data, clustering, clusters, clustering_function=None, n_random=10):

    # Individual
    individual = {}
    individual["Scores_Silhouette"] = sklearn.metrics.silhouette_samples(data, clusters)

    # General
    general = {}
    general["Score_Silhouette"] = sklearn.metrics.silhouette_score(data, clusters)
    general["Score_Calinski"] = sklearn.metrics.calinski_harabasz_score(data, clusters)
    general["Score_Bouldin"] = sklearn.metrics.davies_bouldin_score(data, clusters)

    # Variance explained
    general["Score_VarianceExplained"] = _cluster_quality_variance(data, clusters)

    # Gap statistic
    if clustering_function is not None:
        general["Score_GAP"] = _cluster_quality_gap(data,
                                                    clusters,
                                                    clustering_function,
                                                    n_random=n_random)
    return individual, general



def _cluster_quality_sumsquares(data, clusters):
    """Within-clusters sum of squares
    """
    distance = np.min(scipy.spatial.distance.cdist(data, clusters), axis=1)
    return np.sum(distance**2)


def _cluster_quality_variance(data, clusters):
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
        random_data[:, :, i] = random_data[:, :, i] * scipy.matrix(np.diag(maxs - mins)) + mins
        random_clustering, random_clusters = clustering_function(random_data[:, :, i])
        sum_squares_random[i] = _cluster_quality_sumsquares(random_data[:, :, i], random_clusters)
    gap = np.log(np.mean(sum_squares_random))-np.log(sum_squares_within)
    return gap

