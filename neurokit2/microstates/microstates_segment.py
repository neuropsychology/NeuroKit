# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings

from ..stats import mad, standardize
from ..eeg import eeg_gfp
from .microstates_peaks import microstates_peaks
from .microstates_prepare_data import _microstates_prepare_data
from .microstates_quality import microstates_gev
from .microstates_classify import microstates_classify


def microstates_segment(eeg, n_microstates=4, train="gfp", sampling_rate=None, standardize_eeg=False, n_runs=10, max_iterations=1000, seed=None, **kwargs):
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
    seed : int | numpy.random.RandomState | None
        The seed or ``RandomState`` for the random number generator. Defaults
        to ``None``, in which case a different seed is chosen each time this
        function is called.

    Returns
    -------
    maps : ndarray, shape (n_channels, n_states)
        The topographic maps of the found unique microstates.
    segmentation : ndarray, shape (n_samples,)
        For each sample, the index of the microstate to which the sample has
        been assigned.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> eeg = nk.mne_data("filt-0-40_raw").filter(1, 35)
    >>> eeg = nk.eeg_rereference(eeg, 'average')
    >>>
    >>> out = nk.microstates_segment(eeg)
    >>> nk.microstates_plot(out, gfp=out["GFP"][0:500])
    >>>
    >>> out = nk.microstates_segment(eeg)
    >>> nk.microstates_plot(out, gfp=out["GFP"][0:500])


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
                                                         **kwargs)

    # Normalizing constant (used later for GEV)
    gfp_sum_sq = np.sum(gfp**2)

    # Do several runs of the k-means algorithm, keep track of the best segmentation.
    best_gev = 0
    best_microstates = None
    best_segmentation = None
    for i in range(n_runs):
        microstates = _modified_kmeans_cluster_marjin(data[:, indices],
                                                      n_microstates=n_microstates,
                                                      max_iterations=max_iterations,
                                                      threshold=1e-6,
                                                      seed=seed)

        segmentation = _modified_kmeans_predict(data, microstates)

        # Save iteration with highest global explained variance (GEV)
        gev = microstates_gev(data, microstates, segmentation, gfp_sum_sq)

        if gev > best_gev:
            best_gev, best_microstates, best_segmentation = gev, microstates, segmentation

    # Prepare output
    out = {"Microstates": best_microstates,
           "Sequence": best_segmentation,
           "GEV": best_gev,
           "GFP": gfp,
           "Info": info}

    # Reorder
    out = microstates_classify(out)

    return out





# =============================================================================
# Clustering algorithms
# =============================================================================
def _modified_kmeans_predict(data, microstates):
    """Back-fit kmeans clustering on data
    """
    activation = microstates.dot(data)
    segmentation = np.argmax(np.abs(activation), axis=0)
    return segmentation


def _modified_kmeans_cluster_marjin(data, n_microstates=4, max_iterations=1000, threshold=1e-6, seed=None):
    """The modified K-means clustering algorithm, as implemented by Marijn van Vliet

    https://github.com/wmvanvliet/mne_microstates/blob/master/microstates.py

    Parameters
    -----------
    threshhold : float
        The threshold of convergence for the k-means algorithm, based on
        relative change in noise variance. Defaults to 1e-6.
    """
    if not isinstance(seed, np.random.RandomState):
        seed = np.random.RandomState(seed)
    n_channels, n_samples = data.shape

    # Cache this value for later
    data_sum_sq = np.sum(data ** 2)

    # Select random timepoints for our initial topographic maps
    init_times = seed.choice(n_samples, size=n_microstates, replace=False)
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
            states[state] = data[:, idx].dot(activation[state, idx])
            states[state] /= np.linalg.norm(states[state])

        # Estimate residual noise
        act_sum_sq = np.sum(np.sum(states[segmentation].T * data, axis=0) ** 2)
        residual = np.abs(data_sum_sq - act_sum_sq)
        residual /= np.float(n_samples * (n_channels - 1))

        # Next iteration
        prev_residual = residual
        i += 1

    if i == max_iterations:
        warnings.warn("Modified K-means algorithm failed to converge after " + str(i) + " ",
                      "iterations. Consider increasing 'max_iterations'.")
    return states




def _modified_kmeans_cluster_frederic(data, n_microstates=4, n_runs=10, max_iterations=1000, max_error=1e-6):
    """The modified K-means clustering algorithm, as implemented by von Wagner et al. (2017)

    https://github.com/Frederic-vW/eeg_microstates/blob/master/eeg_microstates.py

    Args:
        data: numpy.array, size = number of EEG channels
        n_maps: number of microstate maps
        n_runs: number of K-means runs (optional)
        maxerr: maximum error for convergence (optional)
        maxiter: maximum number of iterations (optional)
        doplot: plot the results, default=False (optional)
    Returns:
        maps: microstate maps (number of maps x number of channels)
        L: sequence of microstate labels
        gfp_peaks: indices of local GFP maxima
        gev: global explained variance (0..1)
        cv: value of the cross-validation criterion
    """

    n_channels, n_samples = data.shape
    data = data - data.mean(axis=1, keepdims=True)

    # Get local maxima of 1D-array
    def locmax(x):
        dx = np.diff(x) # discrete 1st derivative
        zc = np.diff(np.sign(dx)) # zero-crossings of dx
        m = 1 + np.where(zc == -2)[0] # indices of local max.
        return m

    # GFP peaks
    gfp = np.std(data, axis=1)
    gfp_peaks = locmax(gfp)  # sample points of gfp peaks
    gfp_values = gfp[gfp_peaks]  # values of gfp peaks
    gfp2 = np.sum(gfp_values**2)  # normalizing constant in GEV
    n_gfp = gfp_peaks.shape[0]

    # clustering of GFP peak maps only
    V = data[gfp_peaks, :]
    sumV2 = np.sum(V**2)

    # store results for each k-means run
    cv_list = []  # cross-validation criterion for each k-means run
    gev_list = []  # GEV of each map for each k-means run
    gevT_list = []  # total GEV values for each k-means run
    maps_list = []  # microstate maps for each k-means run
    L_list = []  # microstate label sequence for each k-means run

    for run in range(n_runs):
        # initialize random cluster centroids (indices w.r.t. n_gfp)
        rndi = np.random.permutation(n_gfp)[:n_microstates]
        maps = V[rndi, :]
        # normalize row-wise (across EEG channels)
        maps /= np.sqrt(np.sum(maps**2, axis=1, keepdims=True))
        # initialize
        n_iter = 0
        var0 = 1.0
        var1 = 0.0
        # convergence criterion: variance estimate (step 6)
        while ( (np.abs((var0-var1)/var0) > max_error) & (n_iter < max_iterations) ):
            # (step 3) microstate sequence (= current cluster assignment)
            C = np.dot(V, maps.T)
            C /= (n_samples*np.outer(gfp[gfp_peaks], np.std(maps, axis=1)))
            L = np.argmax(C**2, axis=1)
            # (step 4)
            for k in range(n_microstates):
                Vt = V[L==k, :]
                # (step 4a)
                Sk = np.dot(Vt.T, Vt)
                # (step 4b)
                evals, evecs = np.linalg.eig(Sk)
                v = evecs[:, np.argmax(np.abs(evals))]
                maps[k, :] = v/np.sqrt(np.sum(v**2))
            # (step 5)
            var1 = var0
            var0 = sumV2 - np.sum(np.sum(maps[L, :]*V, axis=1)**2)
            var0 /= (n_gfp*(n_samples-1))
            n_iter += 1
        if (n_iter < max_iterations):
            print("\t\tK-means run {:d}/{:d} converged after {:d} iterations.".format(run+1, n_runs, n_iter))
        else:
            print("\t\tK-means run {:d}/{:d} did NOT converge after {:d} iterations.".format(run+1, n_runs, max_iterations))

        # CROSS-VALIDATION criterion for this run (step 8)
        C_ = np.dot(data, maps.T)
        C_ /= (n_samples*np.outer(gfp, np.std(maps, axis=1)))
        L_ = np.argmax(C_**2, axis=1)
        var = np.sum(data**2) - np.sum(np.sum(maps[L_, :]*data, axis=1)**2)
        var /= (n_channels*(n_samples-1))
        cv = var * (n_samples-1)**2/(n_samples-n_microstates-1.)**2

        # GEV (global explained variance) of cluster k
        gev = np.zeros(n_microstates)
        for k in range(n_microstates):
            r = L == k
            gev[k] = np.sum(gfp_values[r]**2 * C[r, k]**2)/gfp2
        gev_total = np.sum(gev)

        # store
        cv_list.append(cv)
        gev_list.append(gev)
        gevT_list.append(gev_total)
        maps_list.append(maps)
        L_list.append(L_)

    # select best run
    k_opt = np.argmin(cv_list)
    # k_opt = np.argmax(gevT_list)
    maps = maps_list[k_opt]
    # ms_gfp = ms_list[k_opt] # microstate sequence at GFP peaks
    gev = gev_list[k_opt]
    L_ = L_list[k_opt]

    # Plot
#    if doplot:
#        plt.ion()
#        # matplotlib's perceptually uniform sequential colormaps:
#        # magma, inferno, plasma, viridis
#        cm = plt.cm.magma
#        fig, axarr = plt.subplots(1, n_microstates, figsize=(20,5))
#        fig.patch.set_facecolor('white')
#        for imap in range(n_microstates):
#            axarr[imap].imshow(eeg2map(maps[imap, :]), cmap=cm, origin='lower')
#            axarr[imap].set_xticks([])
#            axarr[imap].set_xticklabels([])
#            axarr[imap].set_yticks([])
#            axarr[imap].set_yticklabels([])
#        title = "K-means cluster centroids"
#        axarr[0].set_title(title, fontsize=16, fontweight="bold")
#        plt.show()
#
#        # --- assign map labels manually ---
#        order_str = raw_input("\n\t\tAssign map labels (e.g. 0, 2, 1, 3): ")
#        order_str = order_str.replace(",", "")
#        order_str = order_str.replace(" ", "")
#        if (len(order_str) != n_microstates):
#            if (len(order_str)==0):
#                print("\t\tEmpty input string.")
#            else:
#                print("\t\tParsed manual input: {:s}".format(", ".join(order_str)))
#                print("\t\tNumber of labels does not equal number of clusters.")
#            print("\t\tContinue using the original assignment...\n")
#        else:
#            order = np.zeros(n_microstates, dtype=int)
#            for i, s in enumerate(order_str):
#                order[i] = int(s)
#            print("\t\tRe-ordered labels: {:s}".format(", ".join(order_str)))
#            # re-order return variables
#            maps = maps[order,:]
#            for i in range(len(L)):
#                L[i] = order[L[i]]
#            gev = gev[order]
#            # Figure
#            fig, axarr = plt.subplots(1, n_microstates, figsize=(20,5))
#            fig.patch.set_facecolor('white')
#            for imap in range(n_microstates):
#                axarr[imap].imshow(eeg2map(maps[imap, :]), cmap=cm, origin='lower')
#                axarr[imap].set_xticks([])
#                axarr[imap].set_xticklabels([])
#                axarr[imap].set_yticks([])
#                axarr[imap].set_yticklabels([])
#            title = "re-ordered K-means cluster centroids"
#            axarr[0].set_title(title, fontsize=16, fontweight="bold")
#            plt.show()
#            plt.ioff()
    # return maps, L_, gfp_peaks, gev, cv
    return maps
