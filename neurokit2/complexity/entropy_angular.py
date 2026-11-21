from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fft
import scipy.optimize
import scipy.spatial

from ..misc import NeuroKitWarning
from .utils_complexity_embedding import complexity_embedding


def entropy_angular(signal, delay=1, dimension=2, show=False, **kwargs):
    """**Angular entropy (AngEn)**

    The Angular Entropy (AngEn) is the name that we use in NeuroKit to refer to the complexity
    method described in Nardelli et al. (2022), referred as comEDA due to its application to EDA
    signal. Following the code released by the authors, it comprises the following steps:

    1. The signal is normalized to the range [0, 1] (min-max scaling), and its phase space is
       reconstructed with a time-delay embedding.
    2. The angular distance between each pair of embedded vectors :math:`Y_i` and :math:`Y_j`
       (:math:`i < j`) is the cosine of the angle between them,
       :math:`d_{ij} = \\frac{Y_i \\cdot Y_j}{\\|Y_i\\| \\|Y_j\\|}`. The vectors at the origin
       (made only of the minimum of the signal), whose angle is undefined, are left out.
    3. The probability density function (PDF) of the distances is estimated with the kernel
       density estimator of Botev et al. (2010), based on linear diffusion processes with a
       Gaussian kernel (whose bandwidth is selected automatically). It is evaluated on *B* points
       evenly spread over the range of the distances, extended by half of it on each side, where
       *B* is the number of bins given by the Sturges rule (:math:`\\lceil \\log_2(n) + 1 \\rceil`
       for *n* distances), rounded up to a power of 2. Its values are then scaled to sum to 1.
    4. AngEn is the quadratic :func:`Rényi entropy <entropy_renyi>` of these probabilities
       :math:`p_i`, divided by its maximum, :math:`\\log(B)`, so that it lies between 0 and 1:

    .. math::

      AngEn = \\frac{-\\log \\sum_{i=1}^{B} p_i^2}{\\log(B)}

    Because of the normalization, AngEn does not change when a constant is added to the signal or
    when it is multiplied by a positive one (e.g., a change of units).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter (Nardelli et
        al., 2022, used the first local minimum of the auto mutual information).
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter (Nardelli et
        al., 2022, used the false nearest neighbours).
    show : bool
        If ``True``, plots the PDF of the angular distances.
    **kwargs : optional
        Other arguments (not used).

    Returns
    --------
    angen : float
        The Angular Entropy (AngEn) of the signal. It is NaN (with a warning) for signals with
        missing values, constant, or too short to give two distinct angular distances.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute the index (``Dimension`` and ``Delay``), the points at which the PDF is
        estimated (``bins``) and the PDF (``pdf``, scaled to sum to 1).

    See Also
    --------
    entropy_renyi, complexity_embedding

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal with Laplace Noise
      signal = nk.signal_simulate(duration=2, frequency=[5, 3], noise=0.1)

      # Compute Angular Entropy
      @savefig p_entropy_angular1.png scale=100%
      angen, info = nk.entropy_angular(signal, delay=1, dimension=3, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      angen

    References
    -----------
    * Nardelli, M., Greco, A., Sebastiani, L., & Scilingo, E. P. (2022). ComEDA: A new tool for
      stress assessment based on electrodermal activity. Computers in Biology and Medicine, 150,
      106144.
    * Botev, Z. I., Grotowski, J. F., & Kroese, D. P. (2010). Kernel density estimation via
      diffusion. The Annals of Statistics, 38(5), 2916-2957.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    signal = np.asarray(signal, dtype=float)
    info = {"Dimension": dimension, "Delay": delay}

    # 1. Phase space reconstruction (time-delay embeddings)
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)

    if not np.all(np.isfinite(signal)) or np.ptp(signal) == 0:
        warn(
            "The signal has missing values or is constant, which makes it impossible to compute AngEn. Returning np.nan",
            category=NeuroKitWarning,
        )
        return np.nan, info

    # Min-max normalization of the signal to [0, 1] (as in the code of Nardelli et al., 2022)
    embedded = (embedded - np.min(signal)) / np.ptp(signal)

    # 2. Angular distances (cosine of the angle) between all the pairs of points in the phase
    # space, leaving out the vectors at the origin (whose angle is undefined)
    embedded = embedded[np.any(embedded > 0, axis=1)]
    cosines = 1 - scipy.spatial.distance.pdist(embedded, "cosine")

    if len(cosines) == 0 or np.ptp(cosines) == 0:
        warn(
            "There are fewer than two distinct angular distances (e.g., the signal is too short), which "
            + "makes it impossible to estimate their distribution and AngEn. Returning np.nan",
            category=NeuroKitWarning,
        )
        return np.nan, info

    # 3. Compute the probability density function (PDF) on a number of points given by the
    # Sturges rule, rounded up to a power of 2
    n_bins = int(2 ** np.ceil(np.log2(np.ceil(np.log2(len(cosines)) + 1))))
    bins, pdf = _entropy_angular_kde(cosines, n_bins)
    pdf = pdf / np.sum(pdf)
    info.update({"bins": bins, "pdf": pdf})

    # 4. Apply the quadratic Rényi entropy to the PDF, and normalize it to the range [0, 1] by its
    # maximum, the log of the number of bins. Note that in the paper (eq. 4 page 4) there is a
    # minus sign, but adding it would give negative values, and the linked code does not do that
    # (https://github.com/NardelliM/ComEDA/blob/main/comEDA.m#L103)
    angen = -np.log(np.sum(pdf**2)) / np.log(n_bins)

    if show is True:
        # Plot the PDF as a bar chart
        plt.bar(bins, pdf, width=bins[1] - bins[0], alpha=0.5)
        plt.xlabel("Angular distance (cosine)")
        plt.ylabel("Probability")
        # Print titles
        plt.suptitle(f"Angular Entropy (AngEn) = {angen:.3f}")
        plt.title("Distribution of Angular Distances")

    return angen, info


def _entropy_angular_kde(x, n):
    """Kernel density estimation via diffusion (Botev et al., 2010), as in the kde.m function used by
    Nardelli et al. (2022). The density of x is estimated on n points (a power of 2) evenly spread
    from min(x) - range / 2 to max(x) + range / 2.
    """
    x_range = np.max(x) - np.min(x)
    mesh = np.linspace(np.min(x) - x_range / 2, np.max(x) + x_range / 2, n)
    width = mesh[-1] - mesh[0]
    # Number of distinct values (of the cosines, rounded so that the values that differ only by
    # floating-point errors, frequent with quantized signals, are counted once)
    n_unique = len(np.unique(np.round(x, 12)))

    # Bin the data on the mesh (each value is counted at the mesh point below it, as histc)
    initial = np.bincount(np.searchsorted(mesh, x, side="right") - 1, minlength=n) / len(x)

    # Discrete cosine transform of the binned data (with the first coefficient halved, as dct1d)
    a = scipy.fft.dct(initial, type=2)
    a[0] /= 2

    # The squared bandwidth (relative to the width of the mesh) is the root of t - xi * gamma(t)
    squares = np.arange(1, n, dtype=float) ** 2
    a2 = (a[1:] / 2) ** 2
    t_star = _entropy_angular_root(lambda t: _entropy_angular_fixedpoint(t, n_unique, squares, a2), n_unique)

    # Smooth the coefficients with the Gaussian kernel and apply the inverse transform (as idct1d)
    a_t = a * np.exp(-(np.arange(n) ** 2) * np.pi**2 * t_star / 2)
    a_t[0] *= 2
    density = n * scipy.fft.idct(a_t, type=2) / width
    # The negative values (from round-off errors, or ringing of the transform when the bandwidth is
    # small relative to the spacing of the mesh) are set to eps, as in kde.m
    density[density < 0] = np.finfo(float).eps
    return mesh, density


def _entropy_angular_fixedpoint(t, n, squares, a2):
    """The function t - xi * gamma^[7](t) of Botev et al. (2010), for n data points."""
    ell = 7
    f = 2 * np.pi ** (2 * ell) * np.sum(squares**ell * a2 * np.exp(-squares * np.pi**2 * t))
    for s in range(ell - 1, 1, -1):
        k0 = np.prod(np.arange(1, 2 * s, 2)) / np.sqrt(2 * np.pi)
        const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
        time = (2 * const * k0 / n / f) ** (2 / (3 + 2 * s))
        f = 2 * np.pi ** (2 * s) * np.sum(squares**s * a2 * np.exp(-squares * np.pi**2 * time))
    return t - (2 * n * np.sqrt(np.pi) * f) ** (-2 / 5)


def _entropy_angular_root(function, n):
    """Root of the function, searched as in kde.m: in [0, tol], doubling tol (up to 0.1) until the
    function changes sign, or else the minimum of its absolute value in [0, 0.1]."""
    n = min(max(n, 50), 1050)
    tol = 10**-12 + 0.01 * (n - 50) / 1000
    while True:
        try:
            return scipy.optimize.brentq(function, 0, tol, xtol=1e-15)
        except ValueError:
            tol = min(tol * 2, 0.1)
        if tol == 0.1:
            return scipy.optimize.fminbound(lambda t: np.abs(function(t)), 0, 0.1)
