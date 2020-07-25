# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.mixture
import sklearn.decomposition
import scipy.spatial
import scipy.linalg
import functools


def cluster(data, method="kmeans", n_clusters=2, random_state=None, **kwargs):
    """
    Performs clustering of data according to different algorithms.

    Parameters
    ----------
    data : np.ndarray
        Matrix array of data (E.g., an array (channels, times) of M/EEG data).
    method : str
        The algorithm for clustering. Can be one of 'kmeans' (default) modified k-means algorithm 'kmod',
        'pca' (Principal Component Analysis), 'ica' (Independent Component Analysis),
        'agglomerative' (Atomize and Agglomerate Hierarchical Clustering), 'hierarchical', 'spectral',
        'mixture', 'mixturebayesian'. See ``sklearn`` for methods details.
    n_clusters : int
        The desired number of clusters.
    random_state : Union[int, numpy.random.RandomState]
        The ``RandomState`` for the random number generator. Defaults to ``None``, in which case a
        different random state is chosen each time this function is called.
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
    >>> import neurokit2 as nk
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # Load the iris dataset
    >>> data = nk.data("iris")
    >>>
    >>> # Cluster using different methods
    >>> clustering_kmeans, clusters_kmeans, info = nk.cluster(data, method="kmeans", n_clusters=3)
    >>> clustering_kmod, clusters_kmod, info = nk.cluster(data, method="kmod", n_clusters=3)
    >>> clustering_spectral, clusters_spectral, info = nk.cluster(data, method="spectral", n_clusters=3)
    >>> clustering_hierarchical, clusters_hierarchical, info = nk.cluster(data, method="hierarchical", n_clusters=3)
    >>> clustering_agglomerative, clusters_agglomerative, info= nk.cluster(data, method="agglomerative", n_clusters=3)
    >>> clustering_mixture, clusters_mixture, info = nk.cluster(data, method="mixture", n_clusters=3)
    >>> clustering_bayes, clusters_bayes, info = nk.cluster(data, method="mixturebayesian", n_clusters=3)
    >>> clustering_pca, clusters_pca, info = nk.cluster(data, method="pca", n_clusters=3)
    >>> clustering_ica, clusters_ica, info = nk.cluster(data, method="ica", n_clusters=3)
    >>>
    >>> # Visualize classification and 'average cluster'
    >>> fig, axes = plt.subplots(ncols=2, nrows=5)
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
    >>> axes[3, 0].scatter(data[:, 2], data[:, 3], c=clustering_pca['Cluster'])
    >>> axes[3, 0].scatter(clusters_pca[:, 2], clusters_pca[:, 3], c='red')
    >>> axes[3, 0].set_title("PCA")
    >>> axes[3, 1].scatter(data[:, 2], data[:, 3], c=clustering_ica['Cluster'])
    >>> axes[3, 1].scatter(clusters_ica[:, 2], clusters_ica[:, 3], c='red')
    >>> axes[3, 1].set_title("ICA")
    >>> axes[4, 0].scatter(data[:, 2], data[:, 3], c=clustering_kmod['Cluster'])
    >>> axes[4, 0].scatter(clusters_kmod[:, 2], clusters_kmod[:, 3], c='red')
    >>> axes[4, 0].set_title("modified K-means")
    """
    # Sanity fixes
    if isinstance(data, pd.DataFrame):
        data = data.values

    method = method.lower()
    # K-means
    if method in ["kmeans", "k", "k-means", "kmean"]:
        out =  _cluster_kmeans(data,
                               n_clusters=n_clusters,
                               random_state=random_state,
                               **kwargs)

    # Modified k-means
    elif method in ["kmods", "kmod", "kmeans modified", "modified kmeans"]:
        out = _cluster_kmod(data, n_clusters=n_clusters,
                            random_state=random_state, **kwargs)

    # PCA
    elif method in ["pca", "principal", "principal component analysis"]:
        out = _cluster_pca(data, n_clusters=n_clusters, random_state=random_state, **kwargs)

    # ICA
    elif method in ["ica", "independent", "independent component analysis"]:
        out = _cluster_pca(data, n_clusters=n_clusters, random_state=random_state, **kwargs)

    # Mixture
    elif method in ["mixture", "mixt"]:
        out =  _cluster_mixture(data,
                               n_clusters=n_clusters,
                               bayesian=False,
                               random_state=random_state,
                               **kwargs)

    # Bayesian
    elif method in ["bayesianmixture", "bayesmixt", "mixturebayesian", "mixturebayes"]:
        out =  _cluster_mixture(data,
                               n_clusters=n_clusters,
                               bayesian=True,
                               random_state=random_state,
                               **kwargs)

    # Others
    else:
        out =  _cluster_sklearn(data,
                                n_clusters=n_clusters,
                                **kwargs)

    return out



# =============================================================================
# Methods
# =============================================================================

def _cluster_kmeans(data, n_clusters=2, random_state=None, **kwargs):
    """K-means clustering algorithm
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


def _cluster_kmod(data, init_times=None, n_clusters=2,
                  max_iterations=1000, threshold=1e-6, random_state=None, **kwargs):
    """The modified K-means clustering algorithm, as adapted from Marijn van Vliet.

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
        Random timepoints to be selected for topographic maps. Defaults to None.
    threshold : float
        The threshold of convergence for the k-means algorithm, based on
        relative change in noise variance. Defaults to 1e-6.
    random_state : int | numpy.random.RandomState | None
        The seed or ``RandomState`` for the random number generator. Defaults
        to ``None``, in which case a different seed is chosen each time this
        function is called.

    Returns
    -------
    states : array
        The topographic maps of the found unique microstates which has a shape of n_channels x n_states

    """
    data = data.T
    n_channels, n_samples = data.shape

    # Cache this value for later
    data_sum_sq = np.sum(data ** 2)

    # Select random timepoints for our initial topographic maps
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    if init_times is None:
        init_times = random_state.choice(n_samples, size=n_clusters, replace=False)

    # Iterations
    states = data[:, init_times].T
    states /= np.linalg.norm(states, axis=1, keepdims=True)  # Normalize the maps

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

    # Get distance, and back fit k-means clustering on data
    prediction = _cluster_getdistance(data.T, states)
    prediction["Cluster"] = prediction.abs().idxmin(axis=1).values
    prediction["Cluster"] = [np.where(prediction.columns == state)[0][0] for state in prediction["Cluster"]]

    # Copy function with given parameters
    clustering_function = functools.partial(_cluster_kmod,
                                            n_clusters=n_clusters,
                                            max_iterations=max_iterations,
                                            threshold=threshold,
                                            random_state=random_state,
                                            **kwargs)

    # Info dump
    info = {"n_clusters": n_clusters,
            "clustering_function": clustering_function,
            "random_state": random_state}

    return prediction, states, info


def _cluster_pca(data, n_clusters=2, random_state=None, **kwargs):
    """Principal Component Analysis (PCA) for clustering.
    """
    # Fit PCA
    pca = sklearn.decomposition.PCA(n_components=n_clusters,
                                    copy=True,
                                    whiten=True,
                                    svd_solver='auto',
                                    random_state=random_state,
                                    **kwargs)
    pca = pca.fit(data)
    # clusters = np.array([pca.components_[state, :] for state in range(n_clusters)])

    # Compute variance explained
#    explained_var = pca.explained_variance_ratio_
#    total_explained_var = np.sum(pca.explained_variance_ratio_)

    # Get distance
    prediction = pca.transform(data)
    prediction = pd.DataFrame(prediction).add_prefix("Loading_")
    prediction["Cluster"] = prediction.abs().idxmax(axis=1).values
    prediction["Cluster"] = [np.where(prediction.columns == state)[0][0] for state in prediction["Cluster"]]

    # Recover states from clustering
    clusters = _cluster_getclusters(data, prediction["Cluster"])

    # Copy function with given parameters
    clustering_function = functools.partial(_cluster_pca,
                                            n_clusters=n_clusters,
                                            random_state=random_state,
                                            **kwargs)

    # Info dump
    info = {"n_clusters": n_clusters,
            "clustering_function": clustering_function,
            "random_state": random_state}

    return prediction, clusters, info


def _cluster_ica(data, n_clusters=2, random_state=None, **kwargs):
    """Independent Component Analysis (ICA) for clustering.
    """
    # Fit ICA
    ica = sklearn.decomposition.FastICA(n_components=n_clusters,
                                        algorithm='parallel',
                                        whiten=True,
                                        fun='exp',
                                        random_state=random_state,
                                        **kwargs)

    ica = ica.fit(data)
#    clusters = np.array([ica.components_[state, :] for state in range(n_clusters)])

    # Get distance
    prediction = ica.transform(data)
    prediction = pd.DataFrame(prediction).add_prefix("Loading_")
    prediction["Cluster"] = prediction.abs().idxmax(axis=1).values
    prediction["Cluster"] = [np.where(prediction.columns == state)[0][0] for state in prediction["Cluster"]]

    # Copy function with given parameters
    clustering_function = functools.partial(_cluster_ica,
                                            n_clusters=n_clusters,
                                            random_state=random_state,
                                            **kwargs)

    # Recover states from clustering
    clusters = _cluster_getclusters(data, prediction["Cluster"])

    # Info dump
    info = {"n_clusters": n_clusters,
            "clustering_function": clustering_function,
            "random_state": random_state}

    return prediction, clusters, info


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

#from sys import stdout
#states, L = _frederic_aahc(data, n_clusters=4)
#
#def _frederic_aahc(data, n_clusters=3, doplot=False):
#    """Atomize and Agglomerative Hierarchical Clustering Algorithm
#    AAHC (Murray et al., Brain Topography, 2008)
#    Args:
#        data: EEG data to cluster, numpy.array (n_samples, n_channels)
#        N_clusters: desired number of clusters
#        doplot: boolean, plot maps
#    Returns:
#        maps: n_maps x n_channels (numpy.array)
#    """
#
#    def extract_row(A, k):
#        v = A[k,:]
#        A_ = np.vstack((A[:k,:],A[k+1:,:]))
#        return A_, v
#
#    def extract_item(A, k):
#        a = A[k]
#        A_ = A[:k] + A[k+1:]
#        return A_, a
#
#    def locmax(x):
#        """Get local maxima of 1D-array
#        Args:
#            x: numeric sequence
#        Returns:
#            m: list, 1D-indices of local maxima
#        """
#        dx = np.diff(x) # discrete 1st derivative
#        zc = np.diff(np.sign(dx)) # zero-crossings of dx
#        m = 1 + np.where(zc == -2)[0] # indices of local max.
#        return m
#
#    #print("\n\t--- AAHC ---")
#    nt, nch = data.shape
#
#    # --- get GFP peaks ---
#    gfp = data.std(axis=1)
#    gfp_peaks = locmax(gfp)
#    #gfp_peaks = gfp_peaks[:100]
#    #n_gfp = gfp_peaks.shape[0]
#    gfp2 = np.sum(gfp**2) # normalizing constant in GEV
#
#    # --- initialize clusters ---
#    maps = data[gfp_peaks,:]
#    # --- store original gfp peaks and indices ---
#    cluster_data = data[gfp_peaks,:]
#    #n_maps = n_gfp
#    n_maps = maps.shape[0]
#    print("\t[+] Initial number of clusters: {:d}\n".format(n_maps))
#
#    # --- cluster indices w.r.t. original size, normalized GFP peak data ---
#    Ci = [[k] for k in range(n_maps)]
#
#    # --- main loop: atomize + agglomerate ---
#    while (n_maps > n_clusters):
#        s = "\r{:s}\r\t\tAAHC > n: {:d} => {:d}".format(80*" ", n_maps, n_maps-1)
#        stdout.write(s); stdout.flush()
#        #print("\n\tAAHC > n: {:d} => {:d}".format(n_maps, n_maps-1))
#
#        # --- correlations of the data sequence with each cluster ---
#        m_x, s_x = data.mean(axis=1, keepdims=True), data.std(axis=1)
#        m_y, s_y = maps.mean(axis=1, keepdims=True), maps.std(axis=1)
#        s_xy = 1.*nch*np.outer(s_x, s_y)
#        C = np.dot(data-m_x, np.transpose(maps-m_y)) / s_xy
#
#        # --- microstate sequence, ignore polarity ---
#        L = np.argmax(C**2, axis=1)
#
#        # --- GEV (global explained variance) of cluster k ---
#        gev = np.zeros(n_maps)
#        for k in range(n_maps):
#            r = L==k
#            gev[k] = np.sum(gfp[r]**2 * C[r,k]**2)/gfp2
#
#        # --- merge cluster with the minimum GEV ---
#        imin = np.argmin(gev)
#        #print("\tre-cluster: {:d}".format(imin))
#
#        # --- N => N-1 ---
#        maps, _ = extract_row(maps, imin)
#        Ci, reC = extract_item(Ci, imin)
#        re_cluster = []  # indices of updated clusters
#        #C_sgn = np.zeros(nt)
#        for k in reC:  # map index to re-assign
#            c = cluster_data[k,:]
#            m_x, s_x = maps.mean(axis=1, keepdims=True), maps.std(axis=1)
#            m_y, s_y = c.mean(), c.std()
#            s_xy = 1.*nch*s_x*s_y
#            C = np.dot(maps-m_x, c-m_y)/s_xy
#            inew = np.argmax(C**2) # ignore polarity
#            #C_sgn[k] = C[inew]
#            re_cluster.append(inew)
#            Ci[inew].append(k)
#        n_maps = len(Ci)
#
#        # --- update clusters ---
#        re_cluster = list(set(re_cluster)) # unique list of updated clusters
#
#        ''' re-clustering by modified mean
#        for i in re_cluster:
#            idx = Ci[i]
#            c = np.zeros(nch) # new cluster average
#            for k in idx: # add to new cluster, polarity according to corr. sign
#                if (C_sgn[k] >= 0):
#                    c += cluster_data[k,:]
#                else:
#                    c -= cluster_data[k,:]
#            c /= len(idx)
#            maps[i] = c
#            #maps[i] = (c-np.mean(c))/np.std(c) # normalize the new cluster
#        del C_sgn
#        '''
#
#        # re-clustering by eigenvector method
#        for i in re_cluster:
#            idx = Ci[i]
#            Vt = cluster_data[idx,:]
#            Sk = np.dot(Vt.T, Vt)
#            evals, evecs = np.linalg.eig(Sk)
#            c = evecs[:, np.argmax(np.abs(evals))]
#            c = np.real(c)
#            maps[i] = c/np.sqrt(np.sum(c**2))
#
#    print()
#    return maps, L

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
