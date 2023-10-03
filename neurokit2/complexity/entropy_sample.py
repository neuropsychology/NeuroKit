# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .optim_complexity_tolerance import complexity_tolerance
from .utils_entropy import _phi, _phi_divide


def entropy_sample(signal, delay=1, dimension=2, tolerance="sd", **kwargs):
    """**Sample Entropy (SampEn)**

    Compute the sample entropy (SampEn) of a signal. SampEn is a modification
    of ApEn used for assessing complexity of physiological time series signals. It corresponds to
    the conditional probability that two vectors that are close to each other for *m* dimensions
    will remain close at the next *m + 1* component.

    This function can be called either via ``entropy_sample()`` or ``complexity_sampen()``.

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
    **kwargs : optional
        Other arguments.

    See Also
    --------
    entropy_shannon, entropy_approximate, entropy_fuzzy, entropy_quadratic

    Returns
    ----------
    sampen : float
        The sample entropy of the single time series.
        If undefined conditional probabilities are detected (logarithm
        of sum of conditional probabilities is ``ln(0)``), ``np.inf`` will
        be returned, meaning it fails to retrieve 'accurate' regularity information.
        This tends to happen for short data segments, increasing tolerance
        levels might help avoid this.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute sample entropy.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=5)

      sampen, parameters = nk.entropy_sample(signal, delay=1, dimension=2)
      sampen

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

    # Compute phi
    info["phi"], _ = _phi(
        signal,
        delay=delay,
        dimension=dimension,
        tolerance=info["Tolerance"],
        approximate=False,
        **kwargs
    )

    return _phi_divide(info["phi"]), info
