# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def microstates_gev(eeg, microstates, segmentation, gfp, **kwargs):
    """Global Explained Variance (GEV)
    """
    # Normalizing constant (used later for GEV)
    if isinstance(gfp, (list, np.ndarray, pd.Series)):
        gfp_sum_sq = np.sum(gfp**2)
    else:
        gfp_sum_sq = gfp

    map_corr = _correlate_vectors(eeg, microstates[segmentation].T)
    gev = np.sum((gfp * map_corr) ** 2) / gfp_sum_sq
    return gev


def microstates_crossvalidation(eeg, microstates, gfp, n_channels, n_samples):
    """Compute Cross-Validation Criterion

    https://github.com/Frederic-vW/eeg_microstates/blob/master/eeg_microstates.py#L518
    """

    cluster = np.dot(eeg.T, microstates.T)
    cluster /= (n_channels*np.outer(gfp, np.std(microstates, axis=1)))
    sum_sq = np.argmax(cluster**2, axis=1)
    var = np.sum(eeg.T**2) - np.sum(np.sum(microstates[sum_sq, :]*eeg.T, axis=1)**2)
    var /= (n_samples*(n_channels-1))
    criterion = var * (n_channels - 1)**2/(n_channels - len(microstates) - 1.)**2

    return criterion


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
