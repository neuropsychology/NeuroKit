# -*- coding: utf-8 -*-
import functools
import warnings

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.spatial
import sklearn.cluster
import sklearn.decomposition
import sklearn.mixture

from ..misc import check_random_state
from .cluster_quality import _cluster_quality_distance


def cluster(data, method="kmeans", n_clusters=2, random_state=None, optimize=False, **kwargs):
    """**Data Clustering**

    Performs clustering of data using different algorithms.

    * **kmod**: Modified k-means algorithm.
    * **kmeans**: Normal k-means.
    * **kmedoids**: k-medoids clustering, a more stable version of k-means.
    * **pca**: Principal Component Analysis.
    * **ica**: Independent Component Analysis.
    * **aahc**: Atomize and Agglomerate Hierarchical Clustering. Computationally heavy.
    * **hierarchical**
    * **spectral**
    * **mixture**
    * **mixturebayesian**

    See ``sklearn`` for methods details.

    Parameters
    ----------
    data : np.ndarray
        Matrix array of data (E.g., an array (channels, times) of M/EEG data).
    method : str
        The algorithm for clustering. Can be one of ``"kmeans"`` (default), ``"kmod"``,
        ``"kmedoids"``, ``"pca"``, ``"ica"``, ``"aahc"``, ``"hierarchical"``, ``"spectral"``,
        ``"mixture"``, ``"mixturebayesian"``.
    n_clusters : int
        The desired number of clusters.
    random_state : Union[int, numpy.random.RandomState]
        The ``RandomState`` for the random number generator. Defaults to ``None``, in which case a
        different random state is chosen each time this function is called.
    optimize : bool
        Optimized method in Poulsen et al. (2018) for the *k*-means modified method.
    **kwargs
        Other arguments to be passed into ``sklearn`` functions.

    Returns
    -------
    clustering : DataFrame
        Information about the distance of samples from their respective clusters.
    clusters : np.ndarray
        Coordinates of cluster centers, which has a shape of n_clusters x n_features.
    info : dict
        Information about the number of clusters, the function and model used for clustering.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk
      import matplotlib.pyplot as plt

      # Load the iris dataset
      data = nk.data("iris").drop("Species", axis=1)

      # Cluster using different methods
      clustering_kmeans, clusters_kmeans, info = nk.cluster(data, method="kmeans", n_clusters=3)
      clustering_spectral, clusters_spectral, info = nk.cluster(data, method="spectral", n_clusters=3)
      clustering_hierarchical, clusters_hierarchical, info = nk.cluster(data, method="hierarchical", n_clusters=3)
      clustering_agglomerative, clusters_agglomerative, info= nk.cluster(data, method="agglomerative", n_clusters=3)
      clustering_mixture, clusters_mixture, info = nk.cluster(data, method="mixture", n_clusters=3)
      clustering_bayes, clusters_bayes, info = nk.cluster(data, method="mixturebayesian", n_clusters=3)
      clustering_pca, clusters_pca, info = nk.cluster(data, method="pca", n_clusters=3)
      clustering_ica, clusters_ica, info = nk.cluster(data, method="ica", n_clusters=3)
      clustering_kmod, clusters_kmod, info = nk.cluster(data, method="kmod", n_clusters=3)
      clustering_kmedoids, clusters_kmedoids, info = nk.cluster(data, method="kmedoids", n_clusters=3)
      clustering_aahc, clusters_aahc, info = nk.cluster(data, method='aahc_frederic', n_clusters=3)

      # Visualize classification and 'average cluster'
      @savefig p_cluster1.png scale=100%
      fig, axes = plt.subplots(ncols=2, nrows=5)
      axes[0, 0].scatter(data.iloc[:,[2]], data.iloc[:,[3]], c=clustering_kmeans['Cluster'])
      axes[0, 0].scatter(clusters_kmeans[:, 2], clusters_kmeans[:, 3], c='red')
      axes[0, 0].set_title("k-means")
      axes[0, 1].scatter(data.iloc[:,[2]], data.iloc[:, [3]], c=clustering_spectral['Cluster'])
      axes[0, 1].scatter(clusters_spectral[:, 2], clusters_spectral[:, 3], c='red')
      axes[0, 1].set_title("Spectral")
      axes[1, 0].scatter(data.iloc[:,[2]], data.iloc[:,[3]], c=clustering_hierarchical['Cluster'])
      axes[1, 0].scatter(clusters_hierarchical[:, 2], clusters_hierarchical[:, 3], c='red')
      axes[1, 0].set_title("Hierarchical")
      axes[1, 1].scatter(data.iloc[:,[2]], data.iloc[:,[3]], c=clustering_agglomerative['Cluster'])
      axes[1, 1].scatter(clusters_agglomerative[:, 2], clusters_agglomerative[:, 3], c='red')
      axes[1, 1].set_title("Agglomerative")
      axes[2, 0].scatter(data.iloc[:,[2]], data.iloc[:,[3]], c=clustering_mixture['Cluster'])
      axes[2, 0].scatter(clusters_mixture[:, 2], clusters_mixture[:, 3], c='red')
      axes[2, 0].set_title("Mixture")
      axes[2, 1].scatter(data.iloc[:,[2]], data.iloc[:,[3]], c=clustering_bayes['Cluster'])
      axes[2, 1].scatter(clusters_bayes[:, 2], clusters_bayes[:, 3], c='red')
      axes[2, 1].set_title("Bayesian Mixture")
      axes[3, 0].scatter(data.iloc[:,[2]], data.iloc[:,[3]], c=clustering_pca['Cluster'])
      axes[3, 0].scatter(clusters_pca[:, 2], clusters_pca[:, 3], c='red')
      axes[3, 0].set_title("PCA")
      axes[3, 1].scatter(data.iloc[:,[2]], data.iloc[:,[3]], c=clustering_ica['Cluster'])
      axes[3, 1].scatter(clusters_ica[:, 2], clusters_ica[:, 3], c='red')
      axes[3, 1].set_title("ICA")
      axes[4, 0].scatter(data.iloc[:,[2]], data.iloc[:,[3]], c=clustering_kmod['Cluster'])
      axes[4, 0].scatter(clusters_kmod[:, 2], clusters_kmod[:, 3], c='red')
      axes[4, 0].set_title("modified K-means")
      axes[4, 1].scatter(data.iloc[:,[2]], data.iloc[:,[3]], c=clustering_aahc['Cluster'])
      axes[4, 1].scatter(clusters_aahc[:, 2], clusters_aahc[:, 3], c='red')
      axes[4, 1].set_title("AAHC (Frederic's method)")
      @suppress
      plt.close()

    References
    -----------
    * Park, H. S., & Jun, C. H. (2009). A simple and fast algorithm for K-medoids
      clustering. Expert systems with applications, 36(2), 3336-3341.

    """
    # Sanity checks
    if isinstance(data, pd.DataFrame):
        data = data.values

    method = method.lower()
    # K-means
    if method in ["kmeans", "k", "k-means", "kmean"]:
        out = _cluster_kmeans(data, n_clusters=n_clusters, random_state=random_state, **kwargs)

    # Modified k-means
    elif method in ["kmods", "kmod", "kmeans modified", "modified kmeans"]:
        out = _cluster_kmod(
            data, n_clusters=n_clusters, random_state=random_state, optimize=optimize, **kwargs
        )
    # K-medoids
    elif method in ["kmedoids", "k-medoids", "k-centers"]:
        out = _cluster_kmedoids(data, n_clusters=n_clusters, random_state=random_state, **kwargs)

    # PCA
    elif method in ["pca", "principal", "principal component analysis"]:
        out = _cluster_pca(data, n_clusters=n_clusters, random_state=random_state, **kwargs)

    # ICA
    elif method in ["ica", "independent", "independent component analysis"]:
        out = _cluster_pca(data, n_clusters=n_clusters, random_state=random_state, **kwargs)

    # Mixture
    elif method in ["mixture", "mixt"]:
        out = _cluster_mixture(
            data, n_clusters=n_clusters, bayesian=False, random_state=random_state, **kwargs
        )

    # Frederic's AAHC
    elif method in ["aahc_frederic", "aahc_eegmicrostates"]:
        out = _cluster_aahc(data, n_clusters=n_clusters, random_state=random_state, **kwargs)

    # Bayesian
    elif method in ["bayesianmixture", "bayesmixt", "mixturebayesian", "mixturebayes"]:
        out = _cluster_mixture(
            data, n_clusters=n_clusters, bayesian=True, random_state=random_state, **kwargs
        )

    # Others
    else:
        out = _cluster_sklearn(data, n_clusters=n_clusters, **kwargs)

    return out


# =============================================================================
# =============================================================================
# # Methods
# =============================================================================
# =============================================================================


# =============================================================================
# Kmeans
# =============================================================================
def _cluster_kmeans(data, n_clusters=2, random_state=None, n_init="auto", **kwargs):
    """K-means clustering algorithm"""

    # Initialize clustering function
    clustering_model = sklearn.cluster.KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=n_init, **kwargs
    )

    # Fit
    clustering = clustering_model.fit_predict(data)

    # Get representatives (identical to _cluster_getclusters(),
    # but why recompute when already included)
    clusters = clustering_model.cluster_centers_

    # Get distance
    prediction = _cluster_quality_distance(data, clusters, to_dataframe=True)
    prediction["Cluster"] = clustering

    # Copy function with given parameters
    clustering_function = functools.partial(
        _cluster_kmeans, n_clusters=n_clusters, random_state=random_state, n_init=n_init, **kwargs
    )

    # Info dump
    info = {
        "n_clusters": n_clusters,
        "clustering_function": clustering_function,
        "sklearn_model": clustering_model,
        "random_state": random_state,
    }

    return prediction, clusters, info


# =============================================================================
# K-medoids
# =============================================================================


def _cluster_kmedoids(data, n_clusters=2, max_iterations=1000, random_state=None, **kwargs):
    """Peforms k-medoids clustering which is based on the most centrally located object in a cluster.
    Less sensitive to outliers than K-means clustering.

    Adapted from https://github.com/rakeshvar/kmedoids/. Original proposed algorithm from Park & Jun (2009).
    """
    # Sanitize
    if isinstance(data, pd.DataFrame):
        data = np.array(data)
    n_samples = data.shape[0]

    # Step 1: Initialize random medoids
    rng = check_random_state(random_state)
    ids_of_medoids = rng.choice(n_samples, n_clusters, replace=False)

    # Find distance between objects to their medoids, can be euclidean or manhatten
    def find_distance(x, y, dist_method="euclidean"):
        if dist_method == "euclidean":
            return np.sqrt(np.sum(np.square(x - y), axis=-1))
        elif dist_method == "manhatten":
            return np.sum(np.abs(x - y), axis=-1)

    individual_points = data[:, None, :]
    medoid_points = data[None, ids_of_medoids, :]
    distance = find_distance(individual_points, medoid_points)

    # Assign each point to the nearest medoid
    segmentation = np.argmin(distance, axis=1)

    # Step 2: Update medoids
    for i in range(max_iterations):
        # Find new medoids
        ids_of_medoids = np.full(n_clusters, -1, dtype=int)
        for i in range(n_clusters):
            indices = np.where(segmentation == i)[0]
            distances = find_distance(data[indices, None, :], data[None, indices, :]).sum(axis=0)
            ids_of_medoids[i] = indices[np.argmin(distances)]

        # Step 3: Reassign objects to medoids
        new_distances = find_distance(data[:, None, :], data[None, ids_of_medoids, :])
        new_assignments = np.argmin(new_distances, axis=1)
        diffs = np.mean(new_assignments != segmentation)
        segmentation = new_assignments

        # If more than 10% of assignments are equal to previous segmentation,
        # go back to step 2 (threshold arbitrary?)
        if diffs <= 0.01:
            break

    # Data points as centroids
    clusters = data[ids_of_medoids]

    # Get prediction
    prediction = _cluster_quality_distance(data, clusters, to_dataframe=True)
    prediction["Cluster"] = segmentation

    # Copy function with given parameters
    clustering_function = functools.partial(
        _cluster_kmedoids,
        n_clusters=n_clusters,
        max_iterations=max_iterations,
        random_state=random_state,
    )

    # Info dump
    info = {
        "n_clusters": n_clusters,
        "clustering_function": clustering_function,
        "random_state": random_state,
        "clusters": clusters,
    }

    return prediction, clusters, info


# =============================================================================
# Modified K-means
# =============================================================================
def _cluster_kmod(
    data,
    n_clusters=4,
    max_iterations=1000,
    threshold=1e-6,
    random_state=None,
    optimize=False,
    **kwargs
):
    """The modified K-means clustering algorithm,

    adapted from Marijn van Vliet and Frederic von Wegner.

    https://github.com/wmvanvliet/mne_microstates
    https://github.com/Frederic-vW/eeg_microstates

    Parameters
    -----------
    n_clusters : int
        The number of unique microstates to find. Defaults to 4.
    max_iterations : int
        The maximum number of iterations to perform in the k-means algorithm.
        Defaults to 1000.
    threshold : float
        The threshold of convergence for the k-means algorithm, based on
        relative change in noise variance. Defaults to 1e-6.
    random_state : Union[int, numpy.random.RandomState, None]
        The seed or ``RandomState`` for the random number generator. Defaults
        to ``None``, in which case a different seed is chosen each time this
        function is called.
    optimized : bool
        To use a new optimized method in https://www.biorxiv.org/content/10.1101/289850v1.full.pdf.
        For the Kmeans modified method. Default to False.
    **kwargs
        Other arguments to be passed into ``sklearn`` functions.

    Returns
    -------
    clustering : DataFrame
        Information about the distance of samples from their respective clusters.
    clusters : np.ndarray
        Coordinates of cluster centers, which has a shape of n_clusters x n_features.
    info : dict
        Information about the number of clusters, the function and model used for clustering.

    """
    n_samples, n_channels = data.shape

    # Cache this value for later to compute residual
    data_sum_sq = np.sum(data**2)

    # Select random timepoints for our initial topographic maps
    rng = check_random_state(random_state)
    init_times = rng.choice(n_samples, size=n_clusters, replace=False)

    # Initialize random cluster centroids
    clusters = data[init_times, :]

    # Normalize row-wise (across EEG channels)
    clusters /= np.linalg.norm(clusters, axis=1, keepdims=True)  # Normalize the maps

    # Initialize iteration
    prev_residual = 0
    for i in range(max_iterations):

        # Step 3: Assign each sample to the best matching microstate
        activation = clusters.dot(data.T)
        segmentation = np.argmax(np.abs(activation), axis=0)

        # Step 4: Recompute the topographic maps of the microstates, based on the
        # samples that were assigned to each state.
        for state in np.arange(n_clusters):

            # Get data fro specific state
            idx = segmentation == state
            data_state = data[idx, :]

            # Sanity check
            if np.sum(idx) == 0:
                clusters[state] = 0
                continue

            # Retrieve map values

            if optimize:
                # Method 2 - optimized segmentation
                state_vals = data_state.T.dot(activation[state, idx])
            else:
                # Method 1 - eighen value
                # step 4a
                Sk = np.dot(data_state.T, data_state)
                # step 4b
                eigen_vals, eigen_vectors = scipy.linalg.eigh(Sk)
                state_vals = eigen_vectors[:, np.argmax(np.abs(eigen_vals))]

            state_vals /= np.linalg.norm(state_vals)  # Normalize Map
            clusters[state, :] = state_vals  # Store map

        # Estimate residual noise (step 5)
        act_sum_sq = np.sum(np.sum(clusters[segmentation, :] * data, axis=1) ** 2)
        residual = np.abs(data_sum_sq - act_sum_sq)
        residual = residual / float(n_samples * (n_channels - 1))

        # Have we converged? Convergence criterion: variance estimate (step 6)
        if np.abs(prev_residual - residual) < (threshold * residual):
            break

        # Next iteration
        prev_residual = residual.copy()

    if i == max_iterations:
        warnings.warn(
            "Modified K-means algorithm failed to converge after " + str(i) + "",
            "iterations. Consider increasing 'max_iterations'.",
        )

    # De-normalize
    clusters_unnormalized = _cluster_getclusters(data, segmentation)
    prediction = _cluster_quality_distance(data, clusters_unnormalized, to_dataframe=True)
    prediction["Cluster"] = segmentation

    # Copy function with given parameters
    clustering_function = functools.partial(
        _cluster_kmod,
        n_clusters=n_clusters,
        max_iterations=max_iterations,
        threshold=threshold,
        random_state=random_state,
        **kwargs
    )

    # Info dump
    info = {
        "n_clusters": n_clusters,
        "clustering_function": clustering_function,
        "random_state": random_state,
        "clusters_normalized": clusters,
        "residual": residual,
    }

    return prediction, clusters_unnormalized, info


# =============================================================================
# PCA
# =============================================================================
def _cluster_pca(data, n_clusters=2, random_state=None, **kwargs):
    """Principal Component Analysis (PCA) for clustering."""
    # Fit PCA
    pca = sklearn.decomposition.PCA(
        n_components=n_clusters,
        copy=True,
        whiten=True,
        svd_solver="auto",
        random_state=random_state,
        **kwargs
    )
    pca = pca.fit(data)
    # clusters = np.array([pca.components_[state, :] for state in range(n_clusters)])

    # Compute variance explained
    #    explained_var = pca.explained_variance_ratio_
    #    total_explained_var = np.sum(pca.explained_variance_ratio_)

    # Get distance
    prediction = pca.transform(data)
    prediction = pd.DataFrame(prediction).add_prefix("Loading_")
    prediction["Cluster"] = prediction.abs().idxmax(axis=1).values
    prediction["Cluster"] = [
        np.where(prediction.columns == state)[0][0] for state in prediction["Cluster"]
    ]

    # Recover states from clustering
    clusters = _cluster_getclusters(data, prediction["Cluster"])

    # Copy function with given parameters
    clustering_function = functools.partial(
        _cluster_pca, n_clusters=n_clusters, random_state=random_state, **kwargs
    )

    # Info dump
    info = {
        "n_clusters": n_clusters,
        "clustering_function": clustering_function,
        "random_state": random_state,
    }

    return prediction, clusters, info


# =============================================================================
# ICA
# =============================================================================
def _cluster_ica(data, n_clusters=2, random_state=None, **kwargs):
    """Independent Component Analysis (ICA) for clustering."""
    # Fit ICA
    ica = sklearn.decomposition.FastICA(
        n_components=n_clusters,
        algorithm="parallel",
        whiten=True,
        fun="exp",
        random_state=random_state,
        **kwargs
    )

    ica = ica.fit(data)
    #    clusters = np.array([ica.components_[state, :] for state in range(n_clusters)])

    # Get distance
    prediction = ica.transform(data)
    prediction = pd.DataFrame(prediction).add_prefix("Loading_")
    prediction["Cluster"] = prediction.abs().idxmax(axis=1).values
    prediction["Cluster"] = [
        np.where(prediction.columns == state)[0][0] for state in prediction["Cluster"]
    ]

    # Copy function with given parameters
    clustering_function = functools.partial(
        _cluster_ica, n_clusters=n_clusters, random_state=random_state, **kwargs
    )

    # Recover states from clustering
    clusters = _cluster_getclusters(data, prediction["Cluster"])

    # Info dump
    info = {
        "n_clusters": n_clusters,
        "clustering_function": clustering_function,
        "random_state": random_state,
    }

    return prediction, clusters, info


# =============================================================================
# SKLEARN
# =============================================================================
def _cluster_sklearn(data, method="spectral", n_clusters=2, **kwargs):
    """Spectral clustering"""
    # Initialize clustering function
    if method in ["spectral"]:
        clustering_model = sklearn.cluster.SpectralClustering(n_clusters=n_clusters, **kwargs)
    elif method in ["hierarchical", "ward"]:
        clustering_model = sklearn.cluster.AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward", **kwargs
        )
    elif method in ["agglomerative", "single"]:
        clustering_model = sklearn.cluster.AgglomerativeClustering(
            n_clusters=n_clusters, linkage="single", **kwargs
        )

    # Fit
    clustering = clustering_model.fit_predict(data)

    # Get representatives
    clusters = _cluster_getclusters(data, clustering)

    # Get distance
    prediction = _cluster_quality_distance(data, clusters, to_dataframe=True)
    prediction["Cluster"] = clustering

    # Else, copy function
    clustering_function = functools.partial(_cluster_sklearn, n_clusters=n_clusters, **kwargs)

    # Info dump
    info = {
        "n_clusters": n_clusters,
        "clustering_function": clustering_function,
        "sklearn_model": clustering_model,
    }

    return prediction, clusters, info


def _cluster_mixture(data, n_clusters=2, bayesian=False, random_state=None, **kwargs):
    """Mixture model"""
    # Initialize clustering function
    if bayesian is False:
        clustering_model = sklearn.mixture.GaussianMixture(
            n_components=n_clusters, random_state=random_state, **kwargs
        )
    else:
        clustering_model = sklearn.mixture.BayesianGaussianMixture(
            n_components=n_clusters, random_state=random_state, **kwargs
        )

    # Fit
    clustering = clustering_model.fit_predict(data)

    # Get representatives
    clusters = clustering_model.means_

    # Get probability
    prediction = clustering_model.predict_proba(data)
    prediction = pd.DataFrame(prediction).add_prefix("Probability_")
    prediction["Cluster"] = clustering

    # Else, copy function
    clustering_function = functools.partial(
        _cluster_mixture, n_clusters=n_clusters, random_state=random_state, **kwargs
    )

    # Info dump
    info = {
        "n_clusters": n_clusters,
        "clustering_function": clustering_function,
        "sklearn_model": clustering_model,
        "random_state": random_state,
    }

    return prediction, clusters, info


# =============================================================================
# AAHC
# =============================================================================


def _cluster_aahc(
    data,
    n_clusters=2,
    gfp=None,
    gfp_peaks=None,
    gfp_sum_sq=None,
    random_state=None,
    use_peaks=False,
    **kwargs
):
    """Atomize and Agglomerative Hierarchical Clustering Algorithm, AAHC (Murray et al., Brain Topography, 2008),
    implemented by https://github.com/Frederic-vW/eeg_microstates/blob/master/eeg_microstates.py#L518

    Preprocessing steps of GFP computation are necessary for the algorithm to run. If gfp arguments are specified,
    data is assumed to have been filtered out based on gfp peaks (e.g., data[:, indices]), if not specified,
    gfp indices will be calculated in the algorithm and data is assumed to be the full un-preprocessed input.
    """

    # Internal functions for aahc
    def extract_row(A, k):
        v = A[k, :]
        A_ = np.vstack((A[:k, :], A[k + 1 :, :]))
        return A_, v

    def extract_item(A, k):
        a = A[k]
        A_ = A[:k] + A[k + 1 :]
        return A_, a

    def locmax(x):
        """Get local maxima of 1D-array
        Args:
            x: numeric sequence
        Returns:
            m: list, 1D-indices of local maxima
        """
        dx = np.diff(x)  # discrete 1st derivative
        zc = np.diff(np.sign(dx))  # zero-crossings of dx
        m = 1 + np.where(zc == -2)[0]  # indices of local max.
        return m

    # Sanitize
    if isinstance(data, pd.DataFrame):
        data = np.array(data)
    _, nch = data.shape

    # If preprocessing is not Done already
    if gfp is None and gfp_peaks is None and gfp_sum_sq is None:
        gfp = data.std(axis=1)
        gfp_peaks = locmax(gfp)
        gfp_sum_sq = np.sum(gfp**2)  # normalizing constant in GEV
        if use_peaks:
            maps = data[gfp_peaks, :]  # initialize clusters
            cluster_data = data[gfp_peaks, :]  # store original gfp peak indices
        else:
            maps = data.copy()
            cluster_data = data.copy()
    else:
        maps = data.copy()
        cluster_data = data.copy()

    n_maps = maps.shape[0]

    # cluster indices w.r.t. original size, normalized GFP peak data
    Ci = [[k] for k in range(n_maps)]

    # Main loop: atomize + agglomerate
    while n_maps > n_clusters:

        # correlations of the data sequence with each cluster
        m_x, s_x = data.mean(axis=1, keepdims=True), data.std(axis=1)
        m_y, s_y = maps.mean(axis=1, keepdims=True), maps.std(axis=1)
        s_xy = 1.0 * nch * np.outer(s_x, s_y)
        C = np.dot(data - m_x, np.transpose(maps - m_y)) / s_xy

        # microstate sequence, ignore polarity
        L = np.argmax(C**2, axis=1)

        # GEV (global explained variance) of cluster k
        gev = np.zeros(n_maps)
        for k in range(n_maps):
            r = L == k
            gev[k] = np.sum(gfp[r] ** 2 * C[r, k] ** 2) / gfp_sum_sq

        # merge cluster with the minimum GEV
        imin = np.argmin(gev)

        # N => N-1
        maps, _ = extract_row(maps, imin)
        Ci, reC = extract_item(Ci, imin)
        re_cluster = []  # indices of updated clusters
        for k in reC:  # map index to re-assign
            c = cluster_data[k, :]
            m_x, s_x = maps.mean(axis=1, keepdims=True), maps.std(axis=1)
            m_y, s_y = c.mean(), c.std()
            s_xy = 1.0 * nch * s_x * s_y
            C = np.dot(maps - m_x, c - m_y) / s_xy
            inew = np.argmax(C**2)  # ignore polarity
            re_cluster.append(inew)
            Ci[inew].append(k)
        n_maps = len(Ci)

        # Update clusters
        re_cluster = list(set(re_cluster))  # unique list of updated clusters

        # re-clustering by eigenvector method
        for i in re_cluster:
            idx = Ci[i]
            Vt = cluster_data[idx, :]
            Sk = np.dot(Vt.T, Vt)
            evals, evecs = np.linalg.eig(Sk)
            c = evecs[:, np.argmax(np.abs(evals))]
            c = np.real(c)
            maps[i] = c / np.sqrt(np.sum(c**2))

    # Get distance
    prediction = _cluster_quality_distance(cluster_data, maps, to_dataframe=True)
    prediction["Cluster"] = prediction.abs().idxmax(axis=1).values
    prediction["Cluster"] = [
        np.where(prediction.columns == state)[0][0] for state in prediction["Cluster"]
    ]

    # Function
    clustering_function = functools.partial(
        _cluster_aahc, n_clusters=n_clusters, random_state=random_state, **kwargs
    )

    # Info dump
    info = {
        "n_clusters": n_clusters,
        "clustering_function": clustering_function,
        "random_state": random_state,
    }

    return prediction, maps, info


# =============================================================================
# =============================================================================
# # Utils
# =============================================================================
# =============================================================================


def _cluster_getclusters(data, clustering):
    """Get average representatives of clusters"""
    n_clusters = len(np.unique(clustering))
    return np.asarray([np.mean(data[np.where(clustering == i)], axis=0) for i in range(n_clusters)])
