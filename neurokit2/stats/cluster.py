# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.mixture
import scipy.spatial
import functools


def cluster(data, method="kmeans", n_clusters=2, random_state=None, **kwargs):
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
    >>> clustering_kmeans, clusters_kmeans, info = nk.cluster(data, method="kmeans", n_clusters=3)
    >>> clustering_kmod, clusters_kmod, info = nk.cluster(data, method="kmod", n_clusters=3)
    >>> clustering_spectral, clusters_spectral, info = nk.cluster(data, method="spectral", n_clusters=3)
    >>> clustering_hierarchical, clusters_hierarchical, info = nk.cluster(data, method="hierarchical", n_clusters=3)
    >>> clustering_agglomerative, clusters_agglomerative , info= nk.cluster(data, method="agglomerative", n_clusters=3)
    >>> clustering_mixture, clusters_mixture, info = nk.cluster(data, method="mixture", n_clusters=3)
    >>> clustering_bayes, clusters_bayes, info = nk.cluster(data, method="mixturebayesian", n_clusters=3)
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
    >>> axes[2, 1].scatter(data[:, 2], data[:, 3], c=clustering_bayes['Cluster'])
    >>> axes[2, 1].scatter(clusters_bayes[:, 2], clusters_bayes[:, 3], c='red')
    >>> axes[2, 1].set_title("Bayesian Mixture")
    """
    method = method.lower()
    if method in ["kmeans", "k", "k-means", "kmean"]:
        out =  _cluster_kmeans(data,
                               n_clusters=n_clusters,
                               random_state=random_state,
                               **kwargs)
    elif method in ["kmods", "kmod", "kmeans modified", "modified kmeans"]:
        out = _cluster_kmod(data, n_clusters=n_clusters,
                            random_state=random_state, **kwargs)
    elif method in ["mixture", "mixt"]:
        out =  _cluster_mixture(data,
                               n_clusters=n_clusters,
                               bayesian=False,
                               random_state=random_state,
                               **kwargs)
    elif method in ["bayesianmixture", "bayesmixt", "mixturebayesian", "mixturebayes"]:
        out =  _cluster_mixture(data,
                               n_clusters=n_clusters,
                               bayesian=True,
                               random_state=random_state,
                               **kwargs)
    else:
        out =  _cluster_sklearn(data,
                                n_clusters=n_clusters,
                                **kwargs)

    return out



# =============================================================================
# Methods
# =============================================================================

def _cluster_kmeans(data, n_clusters=2, random_state=None, **kwargs):
    """K-means
    """
    # Initialize clustering function
    clustering_model = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                              random_state=random_state,
                                              **kwargs)

    # Fit
    clustering = clustering_model.fit_predict(data)

    # Get representatives (identical to _cluster_getclusters(), but why recompute when already included)
    clusters = clustering_model.cluster_centers_

    # Get distance
    prediction = _cluster_getdistance(data, clusters)
    prediction["Cluster"] = clustering

    # Copy function with given parameters
    clustering_function = functools.partial(_cluster_kmeans,
                                            n_clusters=n_clusters,
                                            random_state=random_state,
                                            **kwargs)

    # Info dump
    info = {"n_clusters": n_clusters,
            "clustering_function": clustering_function,
            "sklearn_model": clustering_model,
            "random_state": random_state}

    return prediction, clusters, info


def _cluster_kmod(data, init_times=None, n_clusters=4, max_iterations=1000, threshold=1e-6, random_state=None, **kwargs):
    """The modified K-means clustering algorithm, as implemented by Marijn van Vliet.

    https://github.com/wmvanvliet/mne_microstates/blob/master/microstates.py

    Parameters
    -----------
    data : np.ndarray
        An array (channels x times) of MEEG data, obtained from Raw or Epochs object from MNE.
    n_microstates : int
        The number of unique microstates to find. Defaults to 4.
    max_iterations : int
        The maximum number of iterations to perform in the k-means algorithm.
        Defaults to 1000.
    init_times : array
        Random timepoints to be selected for topographic maps.
    threshold : float
        The threshold of convergence for the k-means algorithm, based on
        relative change in noise variance. Defaults to 1e-6.
    seed : int | numpy.random.RandomState | None
        The seed or ``RandomState`` for the random number generator. Defaults
        to ``None``, in which case a different seed is chosen each time this
        function is called.

    Returns
    -------
    states : array
        The topographic maps of the found unique microstates which has a shape of n_channels x n_states

    """
    n_channels, n_samples = data.shape

    # Cache this value for later
    data_sum_sq = np.sum(data ** 2)

    # Select random timepoints for our initial topographic maps
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    if init_times is None:
        init_times = random_state.choice(n_samples, size=n_clusters, replace=False)

    states = data[:, init_times].T
    states /= np.linalg.norm(states, axis=1, keepdims=True)  # Normalize the maps

    prev_residual = np.inf

    # Convergence criterion: variance estimate (step 6)
    i = 0
    prev_residual = 1
    residual = 0
    while ((np.abs((prev_residual - residual) / prev_residual) > threshold) & (i < max_iterations)):
        # Assign each sample to the best matching microstate
        activation = states.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)
        # assigned_activations = np.choose(segmentations, all_activations)

        # Recompute the topographic maps of the microstates, based on the
        # samples that were assigned to each state.
        for state in np.arange(n_clusters):
            idx = (segmentation == state)
            if np.sum(idx) == 0:
                warnings.warn('Some microstates are never activated')
                states[state] = 0
                continue

            # Find largest eigenvector
            cov = data[:, idx].dot(data[:, idx].T)
            _, vec = scipy.linalg.eigh(cov, eigvals=(n_channels-1, n_channels-1))
            states[state] = vec.ravel()
#            specific_state = data[:, idx]  # Filter out specific state
#            states[state] = specific_state.dot(activation[state, idx])
            states[state] /= np.linalg.norm(states[state])

        # Estimate residual noise
        act_sum_sq = np.sum(np.sum(states[segmentation].T * data, axis=0) ** 2)
        residual = np.abs(data_sum_sq - act_sum_sq)
        residual /= np.float(n_samples * (n_channels - 1))

        # Next iteration
        prev_residual = residual
        i += 1

    if i == max_iterations:
        warnings.warn("Modified K-means algorithm failed to converge after " + str(i) + "",
                      "iterations. Consider increasing 'max_iterations'.")

    # Get distance
    prediction = _cluster_getdistance(data.T, states)
    prediction["Cluster"] = segmentation

    # Copy function with given parameters
    clustering_function = functools.partial(_cluster_kmod,
                                            n_clusters=n_clusters,
                                            random_state=random_state,
                                            **kwargs)

    # Info dump
    info = {"n_clusters": n_clusters,
            "clustering_function": clustering_function,
            "random_state": random_state}

    return prediction, states, info



def _cluster_sklearn(data, method="spectral", n_clusters=2, **kwargs):
    """Spectral clustering
    """
    # Initialize clustering function
    if method in ["spectral"]:
        clustering_model = sklearn.cluster.SpectralClustering(n_clusters=n_clusters, **kwargs)
    elif method in ["hierarchical", "ward"]:
        clustering_model = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", **kwargs)
    elif method in ["agglomerative", "single"]:
        clustering_model = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage="single", **kwargs)

    # Fit
    clustering = clustering_model.fit_predict(data)

    # Get representatives
    clusters = _cluster_getclusters(data, clustering)

    # Get distance
    prediction = _cluster_getdistance(data, clusters)
    prediction["Cluster"] = clustering

    # Else, copy function
    clustering_function = functools.partial(_cluster_sklearn,
                                            n_clusters=n_clusters,
                                            **kwargs)

    # Info dump
    info = {"n_clusters": n_clusters,
            "clustering_function": clustering_function,
            "sklearn_model": clustering_model}

    return prediction, clusters, info


def _cluster_mixture(data, n_clusters=2, bayesian=False, random_state=None, **kwargs):
    """Mixture model
    """
    # Initialize clustering function
    if bayesian is False:
        clustering_model = sklearn.mixture.GaussianMixture(n_components=n_clusters,
                                                           random_state=random_state,
                                                           **kwargs)
    else:
        clustering_model = sklearn.mixture.BayesianGaussianMixture(n_components=n_clusters,
                                                                   random_state=random_state,
                                                                   **kwargs)

    # Fit
    clustering = clustering_model.fit_predict(data)

    # Get representatives
    clusters = clustering_model.means_

    # Get probability
    prediction = clustering_model.predict_proba(data)
    prediction = pd.DataFrame(prediction).add_prefix("Probability_")
    prediction["Cluster"] = clustering

    # Else, copy function
    clustering_function = functools.partial(_cluster_mixture,
                                            n_clusters=n_clusters,
                                            random_state=random_state,
                                            **kwargs)

        # Info dump
    info = {"n_clusters": n_clusters,
            "clustering_function": clustering_function,
            "sklearn_model": clustering_model,
            "random_state": random_state}

    return prediction, clusters, info

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
