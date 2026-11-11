from warnings import warn

import numpy as np
import pandas as pd
import scipy.spatial

from ..misc import NeuroKitWarning
from .optim_complexity_tolerance import complexity_tolerance
from .utils_complexity_embedding import complexity_embedding


def entropy_kolmogorov(signal=None, delay=1, dimension=3, tolerance="sd"):
    """**Kolmogorov Entropy (K2 or K2En)**

    Kolmogorov Entropy, also known as metric entropy, is related to Lyapunov exponents. The
    Kolmogorov-Sinai entropy (*K*) of a chaotic system is the mean rate at which it creates
    information, and Grassberger & Procaccia (1983) showed that the order-2 Rényi entropy *K2*
    (or correlation entropy), a lower bound of *K* (and often close to it), can be estimated from
    the :func:`correlation integrals <fractal_correlation>` :math:`C_m(r)`, i.e., the proportion of
    the pairs of embedded vectors (of dimension *m*) closer than the distance *r* (Euclidean
    distance), which behave as :math:`C_m(r) \\sim r^{D_2} e^{-m \\tau K_2}`:

    .. math::

      K_2 = \\frac{1}{\\tau} \\ln \\frac{C_m(r)}{C_{m+1}(r)}

    *K2* is thus in nats per sample (divided by the ``delay`` :math:`\\tau`). It converges to the
    *K2* of the system only for small *r* and large *m* (e.g., to :math:`\\ln(2)` for the logistic
    map), whereas it grows without bound as *r* decreases for a random signal; for given values of
    *r* and *m*, it can be used as an index of complexity.

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
    tolerance : float
        Tolerance (often denoted as *r*), distance to consider two data points as similar. If
        ``"sd"`` (default), will be set to :math:`0.2 * SD_{signal}`. See
        :func:`complexity_tolerance` to estimate the optimal value for this parameter.

    Returns
    --------
    k2 : float
        The Kolmogorov Entropy (K2) of the signal. It is NaN when no pair of vectors is closer
        than the tolerance at dimension *m* + 1.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_shannon, fractal_correlation, entropy_sample

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=5)

      k2, info = nk.entropy_kolmogorov(signal)
      k2

    References
    -----------
    * Grassberger, P., & Procaccia, I. (1983). Estimation of the Kolmogorov entropy from a chaotic
      signal. Physical review A, 28(4), 2591.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    signal = np.asarray(signal, dtype=float)
    if np.any(np.isnan(signal)):
        warn("The signal contains missing values (NaN): K2 is returned as NaN.", category=NeuroKitWarning)
        return np.nan, {"Dimension": dimension, "Delay": delay, "Tolerance": np.nan}

    # Store parameters
    info = {
        "Dimension": dimension,
        "Delay": delay,
        "Tolerance": complexity_tolerance(
            signal,
            method=tolerance,
            dimension=dimension,
            show=False,
        )[0],
    }

    Ci = [
        _correlation_integral(
            signal,
            delay=delay,
            dimension=dimension,
            tolerance=info["Tolerance"],
        )
        for dimension in [dimension, dimension + 1]
    ]

    if Ci[1] == 0 or np.isnan(Ci[1]):
        K2 = np.nan
    else:
        K2 = np.log(Ci[0] / Ci[1]) / delay
    return K2, info


def _correlation_integral(signal, delay=3, dimension=2, tolerance=0.2):
    # Time-delay embedding
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    if len(embedded) < 2:
        return np.nan

    # Correlation integral: proportion of the pairs of vectors (i < j) closer than the tolerance
    # (the identical vectors, at a distance of 0, included)
    return np.mean(scipy.spatial.distance.pdist(embedded, "euclidean") < tolerance)
