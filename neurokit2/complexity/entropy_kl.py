import numpy as np
import pandas as pd
import scipy.spatial
import scipy.special

from .utils_complexity_embedding import complexity_embedding


def entropy_kl(signal, delay=1, dimension=2, norm="euclidean", **kwargs):
    """**Kozachenko-Leonenko (K-L) Differential entropy (KLEn)**

    The Kozachenko-Leonenko (K-L) estimate of the differential entropy is also referred to as the
    *nearest neighbor estimate* of entropy.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    norm : str
        The probability norm used when computing k-nearest neighbour distances. Can be
        ``"euclidean"`` (default) or ``"max"``.
    **kwargs : optional
        Other arguments (not used for now).

    Returns
    --------
    klen : float
        The KL-entropy of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute Differential entropy.

    See Also
    --------
    entropy_differential

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal with Laplace Noise
      signal = nk.signal_simulate(duration=2, frequency=5, noise=0.1)

      # Compute Kozachenko-Leonenko (K-L) Entropy
      klen, info = nk.entropy_kl(signal, delay=1, dimension=3)
      klen


    References
    -----------
    * Gautama, T., Mandic, D. P., & Van Hulle, M. M. (2003, April). A differential entropy based
      method for determining the optimal embedding parameters of a signal. In 2003 IEEE
      International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings.
      (ICASSP'03). (Vol. 6, pp. VI-29). IEEE.
    * Beirlant, J., Dudewicz, E. J., Györfi, L., & Van der Meulen, E. C. (1997). Nonparametric
      entropy estimation: An overview. International Journal of Mathematical and Statistical
      Sciences, 6(1), 17-39.
    * Kozachenko, L., & Leonenko, N. (1987). Sample estimate of the entropy of a random vector.
      Problemy Peredachi Informatsii, 23(2), 9-16.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    info = {"Dimension": dimension, "Delay": delay}

    # Time delay embedding
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)

    n, d = embedded.shape

    # Get distance to nearest neighbor for each delay vector
    # -------------------------------------------------------

    # Using cKDTree is a much faster version than:
    # # Euclidean distance between vectors
    # dist = sklearn.metrics.DistanceMetric.get_metric(norm)
    # dist = dist.pairwise(embedded)
    # # Enforce non-zero
    # dist[np.isclose(dist, 0)] = np.nan
    # # pj is the Euclidean distance of the j-th delay vector to its nearest neighbor
    # nearest = np.nanmin(dist, axis=1)

    if norm == "max":  # max norm
        p = np.inf
        log_c_d = 0  # volume of the d-dimensional unit ball
    elif norm == "euclidean":  # euclidean norm
        p = 2
        log_c_d = (d / 2.0) * np.log(np.pi) - np.log(scipy.special.gamma(d / 2.0 + 1))
    else:
        raise ValueError("'norm' not recognized.")

    kdtree = scipy.spatial.cKDTree(embedded)

    # Query all points -- k+1 as query point also in initial set
    k = 1  # We want the first nearest neighbour (k = 0 would be itself)
    nearest, _ = kdtree.query(embedded, k + 1, eps=0, p=p)
    nearest = nearest[:, -1]

    # Enforce non-zero distances
    nearest = nearest[nearest > 0]

    # Compute entropy H
    # -------------------------------------------------------
    # (In Gautama (2003), it's not divided by n but it is in Berilant (1997))
    # (the *2 is because 2*radius=diameter)
    klen = np.sum(np.log(n * 2 * nearest) + np.log(2) + np.euler_gamma) / n

    # The above is what I understand from Gautama (2003)'s equation
    # But empirically the following seems more accurate. If someone could clarify / confirm that
    # it's the correct way (or not), that'd be great
    # (Also I don't fully understand the code below)
    # It was used in https://github.com/paulbrodersen/entropy_estimators/continuous.py
    sum_dist = np.sum(np.log(2 * nearest))
    klen = sum_dist * (d / n) - scipy.special.digamma(k) + scipy.special.digamma(n) + log_c_d

    return klen, info
