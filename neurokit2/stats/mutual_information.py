# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.special
import sklearn.neighbors


def mutual_information(x, y, method="varoquaux", bins=256, sigma=1, normalized=True):
    """
    Computes the (normalized) mutual information (MI) between two vectors from a joint histogram. The mutual information
    of two variables is a measure of the mutual dependence between them. More specifically, it quantifies the "amount of
    information" obtained about one variable by observing the other variable.

    Parameters
    ----------
    x, y : list, array or Series
        A vector of values.
    method : str
        Method to use. Can either be 'varoquaux' or 'nolitsa'.
    bins : int
        Number of bins to use while creating the histogram.
    sigma : float
        Sigma for Gaussian smoothing of the joint histogram. Only used if `method=='varoquaux'`.
    normalized : book
        Compute normalised mutual information. Only used if `method=='varoquaux'`.

    Returns
    -------
    float
        The computed similariy measure.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> x = [3, 3, 5, 1, 6, 3]
    >>> y = [5, 3, 1, 3, 4, 5]
    >>>
    >>> nk.mutual_information(x, y, method="varoquaux") #doctest: +ELLIPSIS
    0.23600751227291816
    >>>
    >>> nk.mutual_information(x, y, method="nolitsa") #doctest: +ELLIPSIS
    1.4591479170272448

    References
    ----------
    - Studholme, jhill & jhawkes (1998). "A normalized entropy measure
    of 3-D medical image alignment". in Proc. Medical Imaging 1998,
    vol. 3338, San Diego, CA, pp. 132-143.

    """
    method = method.lower()
    if method in ["varoquaux"]:
        mi = _mutual_information_varoquaux(x, y, bins=bins, sigma=sigma, normalized=normalized)
    elif method in ["shannon", "nolitsa"]:
        mi = _mutual_information_nolitsa(x, y, bins=bins)
    else:
        raise ValueError("NeuroKit error: mutual_information(): 'method' not recognized.")

    return mi


# =============================================================================
# Methods
# =============================================================================
def _mutual_information_varoquaux(x, y, bins=256, sigma=1, normalized=True):
    """
    Based on Gael Varoquaux's implementation: https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429.
    """
    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    scipy.ndimage.gaussian_filter(jh, sigma=sigma, mode="constant", output=jh)

    # compute marginal histograms
    jh = jh + np.finfo(float).eps
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2))) / np.sum(jh * np.log(jh))) - 1
    else:
        mi = np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) - np.sum(s2 * np.log(s2))

    return mi


def _mutual_information_nolitsa(x, y, bins=256):
    """
    Calculate the mutual information between two random variables.

    Calculates mutual information, I = S(x) + S(y) - S(x,y), between two
    random variables x and y, where S(x) is the Shannon entropy.

    Based on the nolitsa package: https://github.com/manu-mannattil/nolitsa/blob/master/nolitsa/delay.py#L72

    """
    p_x = np.histogram(x, bins)[0]
    p_y = np.histogram(y, bins)[0]
    p_xy = np.histogram2d(x, y, bins)[0].flatten()

    # Convert frequencies into probabilities.  Also, in the limit
    # p -> 0, p*log(p) is 0.  We need to take out those.
    p_x = p_x[p_x > 0] / np.sum(p_x)
    p_y = p_y[p_y > 0] / np.sum(p_y)
    p_xy = p_xy[p_xy > 0] / np.sum(p_xy)

    # Calculate the corresponding Shannon entropies.
    h_x = np.sum(p_x * np.log2(p_x))
    h_y = np.sum(p_y * np.log2(p_y))
    h_xy = np.sum(p_xy * np.log2(p_xy))

    return h_xy - h_x - h_y


# =============================================================================
# JUNK
# =============================================================================
def _nearest_distances(X, k=1):
    """
    From https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
    X = array(N,M)
    N = number of points
    M = number of dimensions
    returns the distance to the kth nearest neighbor for every point in X
    """
    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X)
    d, _ = knn.kneighbors(X)  # the first nearest neighbor is itself
    return d[:, -1]  # returns the distance to the kth nearest neighbor


def _entropy(X, k=1):
    """
    Returns the entropy of X. From https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429.

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

    """

    # Distance to kth nearest neighbor
    r = _nearest_distances(X, k)  # squared distances
    n, d = X.shape
    volume_unit_ball = (np.pi ** (0.5 * d)) / scipy.special.gamma(0.5 * d + 1)
    """
    - F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    """
    return (
        d * np.mean(np.log(r + np.finfo(X.dtype).eps))
        + np.log(volume_unit_ball)
        + scipy.special.psi(n)
        - scipy.special.psi(k)
    )
