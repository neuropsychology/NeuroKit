import numpy as np
import pandas as pd

from .entropy_permutation import _entropy_permutation
from .entropy_renyi import entropy_renyi


def entropy_bubble(signal, delay=1, dimension=3, alpha=2, **kwargs):
    """**Bubble Entropy (BubblEn)**

    Introduced by Manis et al. (2017) with the goal of being independent of parameters such as
    *Tolerance* and *Dimension*. Bubble Entropy is based on :func:`permutation entropy <entropy_permutation>`,
    but instead of the ordinal pattern of each embedded vector, it uses the number of swaps that the
    bubble sort algorithm performs to sort it. The :func:`Rényi entropy <entropy_renyi>` (of order
    2) of the distribution of the number of swaps is computed for the dimensions *m* and *m + 1*,
    and Bubble Entropy is their normalized difference:

    .. math::

      BubbEn = \\frac{H_{swaps}^{m+1} - H_{swaps}^{m}}{\\log \\left( \\frac{m + 1}{m - 1} \\right)}

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*), at least 2. See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    alpha : float
        The *alpha* :math:`\\alpha` parameter (default to 2, as in Manis et al., 2017) for
        :func:`Rényi entropy <entropy_renyi>`.
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
        to compute Bubble Entropy.

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
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    if dimension < 2:
        raise ValueError("The dimension of Bubble Entropy should be at least 2.")

    # Store parameters
    info = {"Dimension": dimension, "Delay": delay, "Alpha": alpha}

    H = [
        _entropy_permutation(
            signal,
            dimension=d,
            delay=delay,
            algorithm=entropy_renyi,
            sorting="bubblesort",
            alpha=alpha,
            **kwargs,
        )
        for d in [dimension, dimension + 1]
    ]

    BubbEn = np.diff(H) / np.log((dimension + 1) / (dimension - 1))

    return BubbEn[0], info
