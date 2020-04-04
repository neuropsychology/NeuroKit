# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.special
import scipy.ndimage



def mutual_information(x, y, sigma=1, normalized=True):
    """
    Computes (normalized) mutual information between two vectors from a
    joint histogram. By Gael Varoquaux (https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429).

    Parameters
    ----------
    x, y : list, array or Series
        A vector of values.
    sigma : float
        Sigma for Gaussian smoothing of the joint histogram.
    normalized : book
        Compute normalised mutual information.

    Returns
    -------
    float
        The computed similariy measure

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> x = [3, 3, 5, 1, 6, 3]
    >>> y = [5, 3, 1, 3, 4, 5]
    >>>
    >>> nk.mutual_information(x, y)

    References
    ----------
    - Studholme,  jhill & jhawkes (1998). "A normalized entropy measure
    of 3-D medical image alignment". in Proc. Medical Imaging 1998,
    vol. 3338, San Diego, CA, pp. 132-143.
    """
    bins = (256, 256)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    scipy.ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                 output=jh)

    # compute marginal histograms
    jh = jh + np.finfo(float).eps
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi









# =============================================================================
# Internals
# =============================================================================
def _nearest_distances(X, k=1):
    '''
    From https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
    X = array(N,M)
    N = number of points
    M = number of dimensions
    returns the distance to the kth nearest neighbor for every point in X
    '''
    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X)
    d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
    return d[:, -1] # returns the distance to the kth nearest neighbor



def _entropy(X, k=1):
    ''' Returns the entropy of X.
    From https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429

    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed
    k : int, optional
        number of nearest neighbors for density estimation
    Notes
    ---------
    - Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    - Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    - Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    '''

    # Distance to kth nearest neighbor
    r = _nearest_distances(X, k) # squared distances
    n, d = X.shape
    volume_unit_ball = (np.pi**(.5*d)) / scipy.special.gamma(.5*d + 1)
    '''
    - F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    '''
    return (d*np.mean(np.log(r + np.finfo(X.dtype).eps))
            + np.log(volume_unit_ball) + scipy.special.psi(n) - scipy.special.psi(k))

