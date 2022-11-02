import numpy as np
import pandas as pd

from .utils_complexity_embedding import complexity_embedding


def entropy_svd(signal, delay=1, dimension=2, show=False):
    """**Singular Value Decomposition (SVD) Entropy**

    SVD entropy (SVDEn) can be intuitively seen as an indicator of how many eigenvectors are needed
    for an adequate explanation of the dataset. In other words, it measures feature-richness: the
    higher the SVD entropy, the more orthogonal vectors are required to adequately explain the
    space-state. Similarly to :func:`Fisher Information (FI) <information_fisher>`, it is based on
    the Singular Value Decomposition of the :func:`time-delay embedded <complexity_embedding>` signal.

    See Also
    --------
    information_fisher, complexity_embedding, complexity_delay, complexity_dimension

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
    show : bool
        If True, will plot the attractor.

    Returns
    ----------
    svd : float
        The singular value decomposition (SVD).
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute SVDEn.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=1, frequency=5)

      @savefig p_entropy_svd1.png scale=100%
      svden, info = nk.entropy_svd(signal, delay=5, dimension=3, show=True)
      @suppress
      plt.close()

      svden

    References
    ----------
    * Roberts, S. J., Penny, W., & Rezek, I. (1999). Temporal and spatial complexity measures for
      electroencephalogram based brain-computer interfacing. Medical & biological engineering &
      computing, 37(1), 93-98.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    embedded = complexity_embedding(signal, delay=delay, dimension=dimension, show=show)
    W = np.linalg.svd(embedded, compute_uv=False)  # Compute SVD
    W /= np.sum(W)  # Normalize singular values

    return -1 * sum(W * np.log2(W)), {"Dimension": dimension, "Delay": delay}
