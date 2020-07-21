# -*- coding: utf-8 -*-
import numpy as np
import warnings
from sklearn.decomposition import PCA, FastICA

from .microstates_prepare_data import _microstates_prepare_data
from .microstates_quality import microstates_gev, microstates_crossvalidation
from .microstates_classify import microstates_classify


def microstates_segment(eeg, n_microstates=4, train="gfp", method='marjin', gfp_method='l1', sampling_rate=None,
                        standardize_eeg=False, n_runs=10, max_iterations=1000, criterion='gev', seed=None, **kwargs):
    """Segment a continuous M/EEG signal into microstates using different clustering algorithms.

    Several runs of the clustering algorithm are performed, using different random initializations.
    The run that resulted in the best segmentation, as measured by global explained variance
    (GEV), is used.

    The microstates clustering is typically fitted on the EEG data at the global field power (GFP)
    peaks to maximize the signal to noise ratio and focus on moments of high global neuronal
    synchronization. It is assumed that the topography around a GFP peak remains stable and is at
    its highest signal-to-noise ratio at the GFP peak.

    Parameters
    ----------
    eeg : np.ndarray
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    n_microstates : int
        The number of unique microstates to find. Defaults to 4.
    train : str | int | float
        Method for selecting the timepoints how which to train the clustering algorithm. Can be
        'gfp' to use the peaks found in the Peaks in the global field power. Can be 'all', in which
        case it will select all the datapoints. It can also be a number or a ratio, in which case
        it will select the corresponding number of evenly spread data points. For instance,
        ``train=10`` will select 10 equally spaced datapoints, whereas ``train=0.5`` will select
        half the data. See ``microstates_peaks()``.
    method : str
        The implementation of the k-means modified algorithm, can be 'marjin' (default) or 'frederic'.
    gfp_method : str
        The GFP extraction method, can be either 'l1' (default) or 'l2' to use the L1 or L2 norm.
        See ``nk.eeg_gfp()`` for more details.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    standardize_eeg : bool
        Standardized (z-score) the data across time prior to GFP extraction
        using ``nk.standardize()``.
    n_runs : int
        The number of random initializations to use for the k-means algorithm.
        The best fitting segmentation across all initializations is used.
        Defaults to 10.
    max_iterations : int
        The maximum number of iterations to perform in the k-means algorithm.
        Defaults to 1000.
    criterion : str
        Which criterion to use to choose the best run for k-means algorithm, can be 'gev' (default) which selects
        the best run based on the highest global explained variance, or 'cv' which selects the best run
        based on the lowest cross-validation criterion. See ``nk.microstates_gev()``
        and ``nk.microstates_crossvalidation()`` for more details respectively.
    seed : int | numpy.random.RandomState | None
        The seed or ``RandomState`` for the random number generator. Defaults
        to ``None``, in which case a different seed is chosen each time this
        function is called.

    Returns
    -------
    maps : array
        The topographic maps of the found unique microstates which has a shape of n_channels x n_states
    segmentation : array
        For each sample, the index of the microstate to which the sample has been assigned.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> eeg = nk.mne_data("filt-0-40_raw").filter(1, 35)
    >>> eeg = nk.eeg_rereference(eeg, 'average')
    >>>
    >>> # Modified kmeans
    >>> out_marjin = nk.microstates_segment(eeg, method='marjin')
    >>> nk.microstates_plot(out_marjin, gfp=out_marjin["GFP"][0:500])
    >>>
    >>> out_frederic = nk.microstates_segment(eeg, method='frederic')
    >>> nk.microstates_plot(out_frederic, gfp=out_frederic["GFP"][0:500])
    >>>
    >>> # PCA
    >>> out_pca = nk.microstates_segment(eeg, method='pca', standardize_eeg=True)
    >>> nk.microstates_plot(out_pca, gfp=out_pca["GFP"][0:500])
    >>>
    >>> # ICA
    >>> out_ica = nk.microstates_segment(eeg, method='ica', standardize_eeg=True)
    >>> nk.microstates_plot(out_ica, gfp=out_ica["GFP"][0:500])

    See Also
    --------
    eeg_gfp, microstates_peaks, microstates_gev, microstates_crossvalidation, microstates_classify

    References
    ----------
    - Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995). Segmentation of brain
    electrical activity into microstates: model estimation and validation. IEEE Transactions
    on Biomedical Engineering.

    """
    # Sanitize input
    data, indices, gfp, info = _microstates_prepare_data(eeg,
                                                         train=train,
                                                         sampling_rate=sampling_rate,
                                                         standardize_eeg=standardize_eeg,
                                                         gfp_method=gfp_method,
                                                         **kwargs)

    # Normalizing constant (used later for GEV)
    gfp_sum_sq = np.sum(gfp**2)

    # Do several runs of the k-means algorithm, keep track of the best segmentation.
#    best_gev = 0
#    best_microstates = None
#    best_segmentation = None
    segmentation_list = []
    microstates_list = []
    cv_list = []
    gev_list = []

    # Random timepoints
    if not isinstance(seed, np.random.RandomState):
        seed = np.random.RandomState(seed)

    # Run choice of clustering algorithm
    for i in range(n_runs):
        init_times = seed.choice(len(indices), size=n_microstates, replace=False)
        if method == 'marjin':
            microstates = _modified_kmeans_cluster_marjin(data[:, indices],
                                                          init_times=init_times,
                                                          n_microstates=n_microstates,
                                                          max_iterations=max_iterations,
                                                          threshold=1e-6)
        elif method == 'frederic':
            microstates = _modified_kmeans_cluster_frederic(data[:, indices], init_times=init_times,
                                                            gfp=gfp, indices=indices,
                                                            n_microstates=n_microstates,
                                                            max_iterations=max_iterations,
                                                            threshold=1e-6)
        elif method == 'pca':
            microstates, explained_var, total_explained_var = _pca_cluster(data[:, indices], n_microstates=n_microstates)
            pca_info = {'Explained Variance': explained_var,
                        'Total Explained Variance': total_explained_var}
        elif method == 'ica':
            microstates = _ica_cluster(data[:, indices], n_microstates=n_microstates, max_iterations=max_iterations)

        microstates_list.append(microstates)

        # Predict
        segmentation = _modified_kmeans_predict(data, microstates)
        segmentation_list.append(segmentation)

        # Select best run with highest global explained variance (GEV) or cross-validation criterion
        gev = microstates_gev(data, microstates, segmentation, gfp_sum_sq)
        gev_list.append(gev)

        cv = microstates_crossvalidation(data, microstates, gfp,
                                         n_channels=data.shape[0], n_samples=data.shape[1])
        cv_list.append(cv)

        # Select optimal
        if criterion == 'gev':
            optimal = np.argmax(gev_list)
        elif criterion == 'cv':
            optimal = np.argmin(cv_list)

        best_microstates = microstates_list[optimal]
        best_segmentation = segmentation_list[optimal]
        best_gev = gev_list[optimal]
        best_cv = cv_list[optimal]

#        if gev > best_gev:
#            best_gev, best_microstates, best_segmentation = gev, microstates, segmentation

    # Prepare output
    out = {"Microstates": best_microstates,
           "Sequence": best_segmentation,
           "GEV": best_gev,
           "GFP": gfp,
           "Cross-Validation Criterion": best_cv,
           "Info": info}

    if method == 'pca':
        out.update(pca_info)

    # Reorder
    out = microstates_classify(out)

    return out

#    # select best run
#    k_opt = np.argmin(cv_list)
#    #k_opt = np.argmax(gevT_list)
#    maps = maps_list[k_opt]

# =============================================================================
# Clustering algorithms
# =============================================================================
def _modified_kmeans_predict(data, microstates):
    """Back-fit kmeans clustering on data
    """
    activation = microstates.dot(data)
    segmentation = np.argmax(np.abs(activation), axis=0)
    return segmentation


def _modified_kmeans_cluster_marjin(data, init_times=None,
                                    n_microstates=4, max_iterations=1000, threshold=1e-6):
    """The modified K-means clustering algorithm, as implemented by Marijn van Vliet.

    https://github.com/wmvanvliet/mne_microstates/blob/master/microstates.py

    Parameters
    -----------
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
    maps : array
        The topographic maps of the found unique microstates which has a shape of n_channels x n_states

    """
    n_channels, n_samples = data.shape

    # Cache this value for later
    data_sum_sq = np.sum(data ** 2)

    # Select random timepoints for our initial topographic maps
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
        for state in np.arange(n_microstates):
            idx = (segmentation == state)
            if np.sum(idx) == 0:
                warnings.warn('Some microstates are never activated')
                states[state] = 0
                continue

#            # Find largest eigenvector
#            cov = data[:, idx].dot(data[:, idx].T)
#            _, vec = scipy.linalg.eigh(cov, eigvals=(n_channels-1, n_channels-1))
#            states[state] = vec.ravel()
            specific_state = data[:, idx]  # Filter out specific state
            states[state] = specific_state.dot(activation[state, idx])
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
    return states


def _modified_kmeans_cluster_frederic(data, init_times=None, gfp=None, indices=None,
                                      n_microstates=4, n_runs=10, max_iterations=1000, threshold=1e-6):
    """The modified K-means clustering algorithm, as implemented by von Wagner et al. (2017)

    https://github.com/Frederic-vW/eeg_microstates/blob/master/eeg_microstates.py
    """

    data = data.T
    n_samples, n_channels = data.shape

    # Get GFP info
    gfp_values = gfp[indices]
#    gfp2 = np.sum(gfp_values**2) # normalizing constant in GEV
    n_gfp = indices.shape[0]

    # Cache this value for later
    data_sum_sq = np.sum(data ** 2)

    # Select random timepoints for our initial topographic maps
    states = data[init_times, :]
    states /= np.sqrt(np.sum(states**2, axis=1, keepdims=True))  # normalize row-wise (across EEG channels)

    # Convergence criterion: variance estimate
    n_iter = 0
    prev_residual = 1
    residual = 0

    while ((np.abs((prev_residual - residual) / prev_residual) > threshold) & (n_iter < max_iterations)):
        # (step 3) microstate sequence (= current cluster assignment)
        C = np.dot(data, states.T)
        # Additional step in this algorithm but does not seem to affect labelling
        C /= (n_channels*np.outer(gfp_values, np.std(states, axis=1)))
        segmentation = np.argmax(C**2, axis=1)  # Label each of the len(n_gfp) maps

        # (step 4)
        for state in range(n_microstates):
            specific_state = data[segmentation == state, :]  # Filter out the specific state
            # (step 4a) Find largest eigenvector
            cov = np.dot(specific_state.T, specific_state)
            # (step 4b)
            evals, evecs = np.linalg.eig(cov)
            v = evecs[:, np.argmax(np.abs(evals))]
            states[state, :] = v/np.sqrt(np.sum(v**2))

        # (step 5)
        # Estimate residual noise and next iteration
        residual = prev_residual
        act_sum_sq = np.sum(np.sum(states[segmentation, :] * data, axis=1) ** 2)
        prev_residual = data_sum_sq - act_sum_sq
        prev_residual /= (n_gfp * (n_channels - 1))
        n_iter += 1

    if n_iter == max_iterations:
        warnings.warn("Modified K-means algorithm failed to converge after " + str(n_iter) + "",
                      "iterations. Consider increasing 'max_iterations'.")

    return states


def _pca_cluster(data, n_microstates=4):
    """Run Principal Component Analysis (PCA) for clustering.
    """
    data = data.T
#    data_norm = data - data.mean(axis=1, keepdims=True)
#    data_norm /= data_norm.std(axis=1, keepdims=True)

    # Fit PCA
    pca = PCA(n_components=n_microstates, copy=True, whiten=True, svd_solver='auto')
    pca.fit(data)
    states = np.array([pca.components_[state, :] for state in range(n_microstates)])

    # Compute variance
    explained_var = pca.explained_variance_ratio_
    total_explained_var = np.sum(pca.explained_variance_ratio_)

    return states, explained_var, total_explained_var


def _ica_cluster(data, n_microstates=4, max_iterations=1000):
    """Run Independent Component Analysis (ICA) for clustering.
    """

    data = data.T
#    data_norm = data - data.mean(axis=1, keepdims=True)
#    data_norm /= data_norm.std(axis=1, keepdims=True)

    # Fit ICA
    ica = FastICA(n_components=n_microstates, algorithm='parallel', whiten=True, fun='exp', max_iter=max_iterations)
    ica.fit_transform(data)
    states = np.array([ica.components_[state, :] for state in range(n_microstates)])

    return states


#def _aahc_cluster(data, init_times=None, gfp=None, indices=None,
#                  n_microstates=4, n_runs=10, max_iterations=1000, threshold=1e-6):
#    """The Atomize and Agglomerative Hierarchical Clustering Algorithm, AAHC
#    (Murray et al., Brain Topography, 2008)
#
#    https://github.com/Frederic-vW/eeg_microstates/blob/master/eeg_microstates.py#L401
#
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
#    #print("\n\t--- AAHC ---")
#    data = data[:, indices].T
#    n_samples, n_channels = data.shape
##
##    # --- get GFP peaks ---
##    gfp = data.std(axis=1)
##    gfp_peaks = locmax(gfp)
##    #gfp_peaks = gfp_peaks[:100]
##    #n_gfp = gfp_peaks.shape[0]
#    gfp2 = np.sum(gfp**2) # normalizing constant in GEV
#
##    # --- initialize clusters ---
##    maps = data[gfp_peaks,:]
##    # --- store original gfp peaks and indices ---
##    cluster_data = data[gfp_peaks,:]
#    #n_maps = n_gfp
#    n_initial_clusters = data.shape[0]
#    print("\t[+] Initial number of clusters: {:d}\n".format(n_initial_clusters))
#
#    # --- cluster indices w.r.t. original size, normalized GFP peak data ---
#    cluster_indices = [[state] for state in range(n_microstates)]
#
#    # --- main loop: atomize + agglomerate ---
#    while (n_initial_clusters > n_microstates):
#        from sys import stdout
#        s = "\r{:s}\r\t\tAAHC > n: {:d} => {:d}".format(80*" ", n_initial_clusters, n_initial_clusters-1)
#        stdout.write(s); stdout.flush()
#        #print("\n\tAAHC > n: {:d} => {:d}".format(n_maps, n_maps-1))
#
#        # --- correlations of the data sequence with each cluster ---
#        m_x, s_x = data.mean(axis=1, keepdims=True), data.std(axis=1)
#        m_y, s_y = data.mean(axis=1, keepdims=True), data.std(axis=1)
#        s_xy = 1.*n_channels*np.outer(s_x, s_y)
#        C = np.dot(data-m_x, np.transpose(data-m_y)) / s_xy
#
#        # --- microstate sequence, ignore polarity ---
#        L = np.argmax(C**2, axis=1)
#
#        # --- GEV (global explained variance) of cluster k ---
#        gev = np.zeros(n_initial_clusters)
#        for k in range(n_initial_clusters):
#            r = L == k
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
#    return maps

