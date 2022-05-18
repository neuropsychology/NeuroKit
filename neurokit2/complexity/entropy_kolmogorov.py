import numpy as np
import pandas as pd

from .utils_complexity_embedding import complexity_embedding
from .optim_complexity_tolerance import complexity_tolerance


def entropy_kolmogorov(signal=None, delay=1, dimension=3, tolerance="sd"):
    """**Kolmogorov Entropy (K2 or K2En)**

    Kolmogorov Entropy, also known as metric entropy, is related to Lyapunov exponents.

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
        The Kolmogorov Entropy entropy of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_shannon

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
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

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

    if Ci[1] == 0:
        K2 = np.nan
    else:
        K2 = np.log(Ci[0] / Ci[1]) / delay
    return K2, info


def _correlation_integral(signal, delay=3, dimension=2, tolerance=0.2):
    # Time-delay embedding
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)

    # Compute distance
    n = len(embedded)
    norm = np.zeros((n - 1, n - 1))
    for i in range(n - 1):
        Temp = np.tile(embedded[i, :], (n - i - 1, 1)) - embedded[i + 1 :, :]
        norm[i, i:] = np.linalg.norm(Temp, axis=1)
    norm[norm == 0] = np.inf

    # correlation integrals Ci
    return 2 * np.sum(norm < tolerance) / (n * (n - 1))
