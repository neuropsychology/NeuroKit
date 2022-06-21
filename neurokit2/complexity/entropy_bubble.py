# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .entropy_permutation import _entropy_permutation
from .entropy_renyi import entropy_renyi


def entropy_bubble(signal, delay=1, dimension=3, alpha=2, **kwargs):
    """**Bubble Entropy (BubblEn)**

    Introduced by Manis et al. (2017) with the goal of being independent of parameters such as
    *Tolerance* and *Dimension*. Bubble Entropy is based on :func:`permutation entropy <entropy_permutation>`,
    but uses the bubble sort algorithm for the ordering procedure instead of the number of swaps
    performed for each vector.

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
    alpha : float
        The *alpha* :math:`\\alpha` parameter (default to 1) for :func:`Rényi entropy <entropy_renyi>`).
    **kwargs : optional
        Other arguments.

    See Also
    --------
    complexity_ordinalpatterns, entropy_permutation, entropy_renyi

    Returns
    ----------
    BubbEn : float
        The Bubble Entropy.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute sample entropy.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=5)

      BubbEn, info = nk.entropy_bubble(signal)
      BubbEn

    References
    ----------
    * Manis, G., Aktaruzzaman, M. D., & Sassi, R. (2017). Bubble entropy: An entropy almost free of
      parameters. IEEE Transactions on Biomedical Engineering, 64(11), 2711-2718.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Store parameters
    info = {"Dimension": dimension, "Delay": delay}

    H = [
        _entropy_permutation(
            signal,
            dimension=d,
            delay=delay,
            algorithm=entropy_renyi,
            sorting="bubblesort",
            **kwargs,
        )
        for d in [dimension, dimension + 1]
    ]

    BubbEn = np.diff(H) / np.log((dimension + 1) / (dimension - 1))

    return BubbEn[0], info
