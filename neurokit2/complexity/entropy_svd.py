import numpy as np
import pandas as pd

from .complexity_embedding import complexity_embedding


def entropy_svd(signal, delay=1, dimension=2):
    """Singular Value Decomposition (SVD) Entropy

    SVD entropy (SVDEn) can be intuitively seen as an indicator of how many eigenvectors are needed
    for an adequate explanation of the dataset. In other words, it measures feature-richness: the
    higher the SVD entropy, the more orthogonal vectors are required to adequately explain the
    dataset.

    See Also
    --------
    information_fisher

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common
        to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or
        to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns
        an array with two columns corresponding to the original signal and its delayed (by Tau) version.

    Returns
    ----------
    svd : float
        The singular value decomposition (SVD).
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute SVD.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>>
    >>> svden, info = nk.entropy_svd(signal, delay=10, dimension=3)

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    W = np.linalg.svd(embedded, compute_uv=False)
    W /= np.sum(W)  # normalize singular values

    return -1 * sum(W * np.log2(W)), {"Dimension": dimension, "Delay": delay}
