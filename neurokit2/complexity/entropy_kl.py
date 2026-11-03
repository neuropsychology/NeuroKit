from warnings import warn

import numpy as np
import pandas as pd
import scipy.spatial
import scipy.special

from ..misc import NeuroKitWarning
from .utils_complexity_embedding import complexity_embedding


def entropy_kl(signal, delay=1, dimension=2, norm="euclidean", **kwargs):
    """**Kozachenko-Leonenko (K-L) Differential entropy (KLEn)**

    The Kozachenko-Leonenko (K-L) estimate of the differential entropy is also referred to as the
    *nearest neighbor estimate* of entropy (Kozachenko & Leonenko, 1987). It estimates the
    differential entropy of the distribution of the *N* delay vectors (of dimension *d*) from the
    distance of each of them to its nearest neighbour. It is computed as in Kraskov et al. (2004,
    eq. 20):

    .. math::

      KLEn = \\psi(N) - \\psi(1) + \\log(c_d) + \\frac{d}{N} \\sum_{i=1}^{N} \\log(\\epsilon_i)

    where :math:`\\psi` is the digamma function (:math:`-\\psi(1)` is the Euler-Mascheroni
    constant), :math:`\\epsilon_i` is twice the distance of the *i*-th vector to its nearest
    neighbour, and :math:`c_d` is the volume of the *d*-dimensional ball of diameter 1 for the norm
    used: 1 for the maximum norm, and :math:`\\pi^{d/2} / \\Gamma(1 + d/2) / 2^d` for the Euclidean
    norm.

    KLEn uses the natural logarithm (it is in *nats*). Like the differential entropy (see
    :func:`entropy_differential`), it depends on the units of the signal: multiplying the signal by
    *a* adds :math:`d \\log(|a|)` to KLEn. Vectors that have an identical neighbour (at a distance
    of 0, e.g., in quantized signals), for which the logarithm is not defined, are left out of the
    average of the log-distances (the estimator is meant for continuous values, and is biased for
    strongly quantized signals). KLEn is NaN when all the vectors are (e.g., for a constant
    signal).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        Defaults to 1. See :func:`complexity_delay` to estimate the optimal value for this
        parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). Defaults to 2. See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    norm : str
        The norm used to compute the distances between the delay vectors (and find the nearest
        neighbours). Can be ``"euclidean"`` (default) or ``"max"``.
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

    For white noise, the delay vectors follow a *d*-dimensional standard normal distribution, whose
    differential entropy is :math:`\\frac{d}{2} \\log(2 \\pi e)`.

    .. ipython:: python

      noise = np.random.default_rng(42).normal(size=5000)
      nk.entropy_kl(noise, dimension=2)[0], np.log(2 * np.pi * np.e)


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
    * Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information.
      Physical Review E, 69(6), 066138.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    if norm == "max":  # max norm
        p = np.inf
    elif norm == "euclidean":  # euclidean norm
        p = 2
    else:
        raise ValueError("'norm' not recognized. Can be 'euclidean' or 'max'.")

    info = {"Dimension": dimension, "Delay": delay, "Norm": norm}

    # Time delay embedding
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    n, d = embedded.shape
    if n < 2:
        warn(
            "At least two delay vectors are needed: KLEn cannot be computed and is returned as NaN.",
            category=NeuroKitWarning,
        )
        return np.nan, info

    # Log of the volume of the d-dimensional ball of diameter 1 (the distances are doubled below)
    if norm == "max":
        log_c_d = 0  # Hypercube of side 1
    else:
        log_c_d = (d / 2) * np.log(np.pi) - scipy.special.gammaln(d / 2 + 1) - d * np.log(2)

    # Distance of each delay vector to its nearest neighbour
    # (k + 1 neighbours are queried, as the closest one is the vector itself)
    k = 1
    nearest, _ = scipy.spatial.cKDTree(embedded).query(embedded, k + 1, eps=0, p=p)
    nearest = nearest[:, -1]

    # Leave out the vectors that have an identical neighbour (log(0) is not defined)
    nearest = nearest[nearest > 0]
    if len(nearest) == 0:
        warn(
            "All the delay vectors have an identical neighbour (e.g., the signal is constant): KLEn"
            " cannot be computed and is returned as NaN.",
            category=NeuroKitWarning,
        )
        return np.nan, info

    # Compute entropy H (the *2 is because 2*radius=diameter)
    klen = scipy.special.digamma(n) - scipy.special.digamma(k) + log_c_d + d * np.mean(np.log(2 * nearest))

    return klen, info
