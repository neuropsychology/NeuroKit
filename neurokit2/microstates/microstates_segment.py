# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings
import scipy

from ..stats import mad, standardize
from ..eeg import eeg_gfp
from .microstates_peaks import microstates_peaks
from .microstates_prepare_data import _microstates_prepare_data
from .microstates_quality import microstates_gev, microstates_crossvalidation
from .microstates_classify import microstates_classify


def microstates_segment(eeg, n_microstates=4, train="gfp", method='marjin', gfp_method='l1', sampling_rate=None,
                        standardize_eeg=False, n_runs=10, max_iterations=1000, seed=None, **kwargs):
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
    >>> # Compare methods
    >>> out_marjin = nk.microstates_segment(eeg, method='marjin')
    >>> nk.microstates_plot(out_marjin, gfp=out_marjin["GFP"][0:500])
    >>>
    >>> out_frederic = nk.microstates_segment(eeg, method='frederic')
    >>> nk.microstates_plot(out_frederic, gfp=out_frederic["GFP"][0:500])

    See Also
    --------
    eeg_gfp, microstates_peaks, microstates_gev, microstates_classify

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
    best_gev = 0
    best_microstates = None
    best_segmentation = None

    # Random timepoints
    if not isinstance(seed, np.random.RandomState):
        seed = np.random.RandomState(seed)
    init_times = seed.choice(len(indices), size=n_microstates, replace=False)

    for i in range(n_runs):
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

        segmentation = _modified_kmeans_predict(data, microstates)

        # Save iteration with highest global explained variance (GEV)
        gev = microstates_gev(data, microstates, segmentation, gfp_sum_sq)

        # Compute cross validation criterion
        cv = microstates_crossvalidation(data, microstates, gfp,
                                         n_channels=data.shape[0], n_samples=data.shape[1])

        if gev > best_gev:
            best_gev, best_microstates, best_segmentation = gev, microstates, segmentation

    # Prepare output
    out = {"Microstates": best_microstates,
           "Sequence": best_segmentation,
           "GEV": best_gev,
           "GFP": gfp,
           "Cross-Validation Criterion": cv,
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
