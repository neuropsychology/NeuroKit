# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.spatial
import sklearn.cluster
import sklearn.metrics
import sklearn.mixture
import sklearn.model_selection

from ..misc import check_random_state


def cluster_quality(data, clustering, clusters=None, info=None, n_random=10, random_state=None, **kwargs):
    """**Assess Clustering Quality**

    Compute quality of the clustering using several metrics.

    Parameters
    ----------
    data : np.ndarray
        A matrix array of data (e.g., channels, sample points of M/EEG data)
    clustering : DataFrame
        Information about the distance of samples from their respective clusters, generated from
        :func:`.cluster`.
    clusters : np.ndarray
        Coordinates of cluster centers, which has a shape of n_clusters x n_features, generated
        from :func:`.cluster`.
    info : dict
        Information about the number of clusters, the function and model used for clustering,
        generated from :func:`.cluster`.
    n_random : int
        The number of random initializations to cluster random data for calculating the GAP
        statistic.
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator. See for ``misc.check_random_state`` for further information.
    **kwargs
        Other argument to be passed on, for instance ``GFP`` as ``'sd'`` in microstates.

    Returns
    -------
    individual : DataFrame
        Indices of cluster quality scores for each sample point.
    general : DataFrame
        Indices of cluster quality scores for all clusters.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Load the iris dataset
      data = nk.data("iris").drop("Species", axis=1)

      # Cluster
      clustering, clusters, info = nk.cluster(data, method="kmeans", n_clusters=3)

      # Compute indices of clustering quality
      individual, general = nk.cluster_quality(data, clustering, clusters, info)
      general


    References
    ----------
    * Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters in a
      data set via the gap statistic. Journal of the Royal Statistical Society: Series B
      (Statistical Methodology), 63(2), 411-423.
    * Mohajer, M., Englmeier, K. H., & Schmid, V. J. (2011). A comparison of Gap statistic
      definitions with and without logarithm function. arXiv preprint arXiv:1103.4767.

    """
    # Seed the random generator for reproducible results
    rng = check_random_state(random_state)

    # Sanity checks
    if isinstance(clustering, tuple):
        clustering, clusters, info = clustering

    if isinstance(data, pd.DataFrame):
        data = data.values

    if isinstance(clustering, pd.DataFrame):
        clustering = clustering["Cluster"].values

    individual, general = _cluster_quality_sklearn(data, clustering, clusters)

    # Individual distance from centroid
    distance = _cluster_quality_distance(data, clusters)
    distance = {"Clustering_Distance_" + str(i): distance[:, i] for i in range(distance.shape[1])}
    individual.update(distance)
    individual = pd.DataFrame(individual)

    # Variance explained
    general["Score_VarianceExplained"] = _cluster_quality_variance(data, clusters, clustering)
    general["Score_GEV"], _ = _cluster_quality_gev(data, clusters, clustering, **kwargs)
    general["Score_CrossValidation"] = _cluster_quality_crossvalidation(data, clusters, clustering)

    # Dispersion
    general["Dispersion"] = _cluster_quality_dispersion(data, clustering, **kwargs)

    # Gap statistic
    general.update(_cluster_quality_gap(data, clusters, clustering, info, n_random=n_random, rng=rng))

    # Mixture models
    if "sklearn_model" in info:
        if isinstance(info["sklearn_model"], sklearn.mixture.GaussianMixture):
            general["Score_AIC"] = info["sklearn_model"].aic(data)
            general["Score_BIC"] = info["sklearn_model"].bic(data)
            general["Score_LogLikelihood"] = info["sklearn_model"].score(data)
            sklearn.model_selection.cross_val_score(info["sklearn_model"], data, cv=10)

    general = pd.DataFrame.from_dict(general, orient="index").T
    return individual, general


# =============================================================================
# Utils
# =============================================================================
def _cluster_quality_sklearn(data, clustering, clusters):
    """Metrics from sklearn"""
    n_clusters = len(clusters)

    # Individual scores
    individual = {}
    if n_clusters == 1:
        individual["Clustering_Silhouette"] = np.full(len(clustering), np.nan)
    else:
        individual["Clustering_Silhouette"] = sklearn.metrics.silhouette_samples(data, clustering)

    # General clustering quality scores
    general = {"n_Clusters": n_clusters}
    if n_clusters == 1:
        general["Score_Silhouette"] = np.nan
        general["Score_Calinski"] = np.nan
        general["Score_Bouldin"] = np.nan
    else:
        general["Score_Silhouette"] = sklearn.metrics.silhouette_score(data, clustering)
        general["Score_Calinski"] = sklearn.metrics.calinski_harabasz_score(data, clustering)
        general["Score_Bouldin"] = sklearn.metrics.davies_bouldin_score(data, clustering)

    return individual, general


def _cluster_quality_distance(data, clusters, to_dataframe=False):
    """Distance between samples and clusters"""
    distance = scipy.spatial.distance.cdist(data, clusters)
    if to_dataframe is True:
        distance = pd.DataFrame(distance).add_prefix("Distance_")
    return distance


def _cluster_quality_sumsquares(data, clusters, clustering):
    """Sumsquares of the distance of each data point to its respective cluster"""
    distance = _cluster_quality_distance(data, clusters)
    min_distance = []  # distance from each sample to the centroid of its cluster
    for idx in range(len(data)):
        cluster_identity = clustering[idx]
        min_distance.append(distance[idx, cluster_identity])
    min_distance_squared = [i**2 for i in min_distance]
    return np.sum(min_distance_squared)


def _cluster_quality_dispersion(data, clustering, n_clusters=4):
    """Sumsquares of the distances between samples within each clusters.
    An error measure for a n_clusters cluster where the lower the better.
    Can be used to compare and find the optimal number of clusters.
    """

    dispersion_state = np.full(n_clusters, np.nan)
    for i in range(n_clusters):
        idx = clustering == i
        data_state = data[idx, :]
        state_size = len(data_state)  # number of samples in this cluster
        if state_size > 0:
            # pair-wise distance between members of the same cluster
            distance = scipy.spatial.distance.cdist(data_state, data_state)
            # sumsquares of distances
            dispersion_state[i] = 0.5 * np.nansum(distance**2) / state_size
        else:
            dispersion_state[i] = np.nan

    dispersion = np.nansum(dispersion_state)
    return dispersion


def _cluster_quality_variance(data, clusters, clustering):
    """Variance explained by clustering"""
    sum_squares_within = _cluster_quality_sumsquares(data, clusters, clustering)
    sum_squares_total = np.sum(scipy.spatial.distance.pdist(data) ** 2) / data.shape[0]
    return (sum_squares_total - sum_squares_within) / sum_squares_total


def _cluster_quality_gap(data, clusters, clustering, info, n_random=10, rng=None):
    """GAP statistic and modified GAP statistic by Mohajer (2011).

    The GAP statistic compares the total within intra-cluster variation for different values of k
    with their expected values under null reference distribution of the data.
    """
    dispersion = _cluster_quality_sumsquares(data, clusters, clustering)

    mins, maxs = np.min(data, axis=0), np.max(data, axis=0)
    dispersion_random = np.full(n_random, np.nan)

    for i in range(n_random):

        # Random data
        random_data = rng.uniform(size=data.shape)

        # Rescale random
        m = (maxs - mins) / (np.max(random_data, axis=0) - np.min(random_data, axis=0))
        b = mins - (m * np.min(random_data, axis=0))
        random_data = np.array(m) * random_data + np.array(b)

        # Cluster random
        _, random_clusters, info = info["clustering_function"](random_data)
        random_activation = random_clusters.dot(random_data.T)
        random_clustering = np.argmax(np.abs(random_activation), axis=0)
        dispersion_random[i] = _cluster_quality_sumsquares(
            random_data, random_clusters, random_clustering
        )

    # Compute GAP
    gap = np.mean(np.log(dispersion_random)) - np.log(dispersion)

    # Compute standard deviation
    sd_k = np.sqrt(np.mean((np.log(dispersion_random) - np.mean(np.log(dispersion_random))) ** 2.0))
    s_k = np.sqrt(1.0 + 1.0 / n_random) * sd_k

    # Calculate Gap* statistic by Mohajer (2011)
    gap_star = np.mean(dispersion_random) - dispersion
    sd_k_star = np.sqrt(np.mean((dispersion_random - dispersion) ** 2.0))
    s_k_star = np.sqrt(1.0 + 1.0 / n_random) * sd_k_star

    out = {
        "Score_GAP": gap,
        "Score_GAPmod": gap_star,
        "Score_GAP_sk": s_k,
        "Score_GAPmod_sk": s_k_star,
    }
    return out


def _cluster_quality_crossvalidation(data, clusters, clustering):
    """Cross-validation index

    The original code by https://github.com/Frederic-vW/eeg_microstates/blob/master/eeg_microstates.py#L600
    leads to an error when the denominator is 0.
    """
    n_rows, n_cols = data.shape  # n_sample, n_channel
    var = np.nansum(data**2) - np.nansum(np.nansum(clusters[clustering, :] * data, axis=1) ** 2)
    var /= n_rows * (n_cols - 1)
    denominator = (n_cols - len(clusters) - 1) ** 2
    if np.abs(denominator) > 0:
        cv = var * (n_cols - 1) ** 2 / denominator
    else:
        # warnings.warn(
        #     "Number of columns in data (" + str(n_cols) + ") is smaller "
        #     "than the number of cluster (" + str(len(clusters)) + ") plus 1. "
        #     "Returning the residual noise instead."
        # )
        cv = np.nan

    #    cv = var * (n_cols - 1)**2 / len(clusters)
    return cv


# def _cluster_quality_gev(data, clusters, clustering, sd=None):
#     if sd is None:
#         sd = np.std(data, axis=1)
#
#     # Normalize row-wise (across columns)
#     clusters /= np.sqrt(np.sum(clusters**2, axis=1, keepdims=True))
#     activation = np.dot(data, clusters.T)
#     activation /= (data.shape[1] * np.outer(sd, np.std(clusters, axis=1)))
#
#     gev = np.zeros(len(clusters))
#     for k in range(len(clusters)):
#         idx = (clustering == k)
#         gev[k] = np.sum(sd[idx]**2 * activation[idx, k]**2)
#     gev_total = np.sum(gev) / np.sum(sd ** 2)
#     return gev_total


def _cluster_quality_gev(data, clusters, clustering, sd=None, n_clusters=4):
    """Global Variance Explained (GEV)"""
    if sd is None:
        sd = np.nanstd(data, axis=1)
    map_corr = _correlate_vectors(data.T, clusters[clustering].T)

    gev_all = np.zeros(n_clusters)
    for state in range(n_clusters):
        idx = clustering == state
        gev_all[state] = np.nansum((sd[idx] * map_corr[idx]) ** 2) / np.nansum(sd**2)

    gev = np.nansum(gev_all)
    #    gev = np.sum((sd * map_corr) ** 2) / np.sum(sd**2)
    return gev, gev_all


def _correlate_vectors(A, B, axis=0):
    """Compute pairwise correlation of multiple pairs of vectors.
    Fast way to compute correlation of multiple pairs of vectors without
    computing all pairs as would with corr(A,B). Borrowed from Oli at Stack
    overflow.

    Note the resulting coefficients vary slightly from the ones
    obtained from corr due differences in the order of the calculations.
    (Differences are of a magnitude of 1e-9 to 1e-17 depending of the tested
    data).

    Parameters
    ----------
    A : array
        The first collection of vectors of shape (n, m)
    B : array
        The second collection of vectors of shape (n, m)
    axis : int
        The axis that contains the elements of each vector. Defaults to 0.

    Returns
    -------
    corr : array
        For each pair of vectors, the correlation between them with shape (m, )

    """
    An = A - np.nanmean(A, axis=axis)
    Bn = B - np.nanmean(B, axis=axis)
    An /= np.linalg.norm(An, axis=axis)
    Bn /= np.linalg.norm(Bn, axis=axis)
    return np.nansum(An * Bn, axis=axis)
