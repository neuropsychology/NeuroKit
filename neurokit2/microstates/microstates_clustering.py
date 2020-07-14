# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import zscore
from scipy.signal import find_peaks
from scipy.linalg import eigh
import matplotlib as mpl
from matplotlib import pyplot as plt
import mne
import warnings



def segment(data, n_states=4, n_inits=10, max_iter=1000, thresh=1e-6,
            normalize=False, min_peak_dist=2, max_n_peaks=10000,
            random_state=None, verbose=None):
    """Segment a continuous signal into microstates.
    Peaks in the global field power (GFP) are used to find microstates, using a
    modified K-means algorithm. Several runs of the modified K-means algorithm
    are performed, using different random initializations. The run that
    resulted in the best segmentation, as measured by global explained variance
    (GEV), is used.
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        The data to find the microstates in
    n_states : int
        The number of unique microstates to find. Defaults to 4.
    n_inits : int
        The number of random initializations to use for the k-means algorithm.
        The best fitting segmentation across all initializations is used.
        Defaults to 10.
    max_iter : int
        The maximum number of iterations to perform in the k-means algorithm.
        Defaults to 1000.
    thresh : float
        The threshold of convergence for the k-means algorithm, based on
        relative change in noise variance. Defaults to 1e-6.
    normalize : bool
        Whether to normalize (z-score) the data across time before running the
        k-means algorithm. Defaults to ``False``.
    min_peak_dist : int
        Minimum distance (in samples) between peaks in the GFP. Defaults to 2.
    max_n_peaks : int | None
        Maximum number of GFP peaks to use in the k-means algorithm. Chosen
        randomly. Set to ``None`` to use all peaks. Defaults to 10000.
    random_state : int | numpy.random.RandomState | None
        The seed or ``RandomState`` for the random number generator. Defaults
        to ``None``, in which case a different seed is chosen each time this
        function is called.
    verbose : int | bool | None
        Controls the verbosity.
    Returns
    -------
    maps : ndarray, shape (n_channels, n_states)
        The topographic maps of the found unique microstates.
    segmentation : ndarray, shape (n_samples,)
        For each sample, the index of the microstate to which the sample has
        been assigned.
    References
    ----------
    .. [1] Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995).
           Segmentation of brain electrical activity into microstates: model
           estimation and validation. IEEE Transactions on Biomedical
           Engineering.
    """
    if normalize:
        data = zscore(data, axis=1)

    # Find peaks in the global field power (GFP)
    gfp = np.std(data, axis=0)
    peaks, _ = find_peaks(gfp, distance=min_peak_dist)
    n_peaks = len(peaks)

    # Limit the number of peaks by randomly selecting them
    if max_n_peaks is not None:
        max_n_peaks = min(n_peaks, max_n_peaks)
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        chosen_peaks = random_state.choice(n_peaks, size=max_n_peaks,
                                           replace=False)
        peaks = peaks[chosen_peaks]

    # Cache this value for later
    gfp_sum_sq = np.sum(gfp ** 2)

    # Do several runs of the k-means algorithm, keep track of the best
    # segmentation.
    best_gev = 0
    best_maps = None
    best_segmentation = None
    for _ in range(n_inits):
        maps = _modified_kmeans(data[:, peaks], n_states, n_inits, max_iter, thresh,
                           random_state, verbose)
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)
        map_corr = _correlate_vectors(data, maps[segmentation].T)

        # Compare across iterations using global explained variance (GEV) of
        # the found microstates.
        gev = sum((gfp * map_corr) ** 2) / gfp_sum_sq
        if gev > best_gev:
            best_gev, best_maps, best_segmentation = gev, maps, segmentation

    return best_maps, best_segmentation


# =============================================================================
# Clustering algorithms
# =============================================================================

def _modified_kmeans(data, n_states=4, n_inits=10, max_iter=1000, thresh=1e-6, random_state=None, verbose=None):
    """The modified K-means clustering algorithm.
    See :func:`segment` for the meaning of the parameters and return
    values.
    """
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    n_channels, n_samples = data.shape

    # Cache this value for later
    data_sum_sq = np.sum(data ** 2)

    # Select random timepoints for our initial topographic maps
    init_times = random_state.choice(n_samples, size=n_states, replace=False)
    maps = data[:, init_times].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)  # Normalize the maps

#    prev_residual = np.inf
    for iteration in range(max_iter):
        # Assign each sample to the best matching microstate
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)
        # assigned_activations = np.choose(segmentations, all_activations)

        # Recompute the topographic maps of the microstates, based on the
        # samples that were assigned to each state.
        for state in range(n_states):
            idx = (segmentation == state)
            if np.sum(idx) == 0:
                warnings.warn('Some microstates are never activated')
                maps[state] = 0
            # Find largest eigenvector
            # cov = data[:, idx].dot(data[:, idx].T)
            # _, vec = eigh(cov, eigvals=(n_channels - 1, n_channels - 1))
            # maps[state] = vec.ravel()
            maps[state] = data[:, idx].dot(activation[state, idx])
            maps[state] /= np.linalg.norm(maps[state])

        # Estimate residual noise
        act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)
        residual = abs(data_sum_sq - act_sum_sq)
        residual /= float(n_samples * (n_channels - 1))

#        # Have we converged?
#        if (prev_residual - residual) > (thresh * residual):
#            warnings.warn('The algorihthm did not converge!')
#        prev_residual = residual
    else:
        warnings.warn('Modified K-means algorithm failed to converge.')

    return maps


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
    A : ndarray, shape (n, m)
        The first collection of vectors
    B : ndarray, shape (n, m)
        The second collection of vectors
    axis : int
        The axis that contains the elements of each vector. Defaults to 0.
    Returns
    -------
    corr : ndarray, shape (m,)
        For each pair of vectors, the correlation between them.
    """
    An = A - np.mean(A, axis=axis)
    Bn = B - np.mean(B, axis=axis)
    An /= np.linalg.norm(An, axis=axis)
    Bn /= np.linalg.norm(Bn, axis=axis)
    return np.sum(An * Bn, axis=axis)