# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings

from ..stats import mad, standardize
from ..eeg import eeg_gfp
from .microstates_peaks import microstates_peaks
from .microstates_prepare_data import _microstates_prepare_data
from .microstates_quality import microstates_gev


def microstates_segment(eeg, n_microstates=4, train="gfp", sampling_rate=None, standardize_eeg=False, n_runs=10, max_iterations=1000, seed=None, **kwargs):
    """Segment a continuous M/EEG signal into microstates using different clustering algorithms.

    Several runs of the clustering algorithm are performed, using different random initializations.
    The run that resulted in the best segmentation, as measured by global explained variance
    (GEV), is used.

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
                                                         select=select,
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
        microstates = _modified_kmeans_cluster(data[:, indices],
                                               n_microstates=n_microstates,
                                               max_iterations=max_iterations,
                                               threshold=1e-6,
                                               seed=seed)

        segmentation = _modified_kmeans_predict(data, microstates)

        # Save iteration with highest global explained variance (GEV)
        gev = microstates_gev(data, microstates, segmentation, gfp_sum_sq)

        if gev > best_gev:
            best_gev, best_microstates, best_segmentation = gev, microstates, segmentation

    out = {"Microstates": best_microstates,
           "Sequence": best_segmentation,
           "GEV": best_gev,
           "GFP": gfp,
           "Info": info}

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


def _modified_kmeans_cluster(data, n_microstates=4, max_iterations=1000, threshold=1e-6, seed=None):
    """The modified K-means clustering algorithm.

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


