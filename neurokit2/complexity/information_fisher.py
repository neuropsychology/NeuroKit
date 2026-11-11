import numpy as np
import pandas as pd

from .utils_complexity_embedding import complexity_embedding


def fisher_information(signal, delay=1, dimension=2):
    """**Fisher Information (FI)**

    The Fisher information was introduced by R. A. Fisher in 1925, as a measure of "intrinsic
    accuracy" in statistical estimation theory. It is central to many statistical fields far beyond
    that of complexity theory. It measures the amount of information that an observable random
    variable carries about an unknown parameter. In complexity analysis, the amount of information
    that a system carries "about itself" is measured. Similarly to :func:`SVDEn <entropy_svd>`, it
    is based on the Singular Value Decomposition (SVD) of the :func:`time-delay embedded <complexity_embedding>`
    signal. The value of FI is usually anti-correlated with other measures of complexity (the more
    information a system withholds about itself, and the more predictable and thus, less complex it
    is).

    The singular values :math:`\\sigma_i` of the embedding matrix, in decreasing order, are
    normalized to sum to 1 (:math:`\\bar{\\sigma}_i = \\sigma_i / \\sum_j \\sigma_j`), as for SVDEn
    (Roberts et al., 1999). FI measures how steeply this singular spectrum falls (James & Lowe,
    2003; Bao et al., 2011, Eq. 10):

    .. math::

      FI = \\sum_{i=1}^{M-1} \\frac{(\\bar{\\sigma}_{i+1} - \\bar{\\sigma}_i)^2}{\\bar{\\sigma}_i}

    where *M* is the embedding dimension. Each term is at most :math:`\\bar{\\sigma}_i`, so FI lies
    between 0 (a flat spectrum, e.g., white noise) and 1 (a single non-zero singular value, e.g., a
    constant signal). It does not depend on the scale of the signal, but it does on its mean, since
    the embedding matrix is not centered: a non-zero mean adds to the first singular value, which
    increases FI. Center (or standardize) the signal first.

    See Also
    --------
    entropy_svd, information_mutual, complexity_embedding, complexity_delay, complexity_dimension

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

    Returns
    -------
    fi : float
        The computed fisher information measure.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute fisher information.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=5)
      fi, info = nk.fisher_information(signal, delay=10, dimension=3)
      fi

    References
    ----------
    * Bao, F. S., Liu, X., & Zhang, C. (2011). PyEEG: An open source Python module for EEG/MEG
      feature extraction. Computational Intelligence and Neuroscience, 2011, 406391.
    * James, C. J., & Lowe, D. (2003). Extracting multisource brain activity from a single
      electromagnetic channel. Artificial Intelligence in Medicine, 28(1), 89-104.
    * Roberts, S. J., Penny, W., & Rezek, I. (1999). Temporal and spatial complexity measures for
      electroencephalogram based brain-computer interfacing. Medical & Biological Engineering &
      Computing, 37(1), 93-98.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    # Singular values, in decreasing order, normalized to sum to 1. The embedding matrix is not
    # centered, so a non-zero mean adds to the first one
    W = np.linalg.svd(embedded, compute_uv=False)
    W /= np.sum(W)
    # One term per pair of consecutive singular values (Bao et al., 2011, Eq. 10). Each is at most
    # W[i], so FI lies between 0 (flat spectrum) and 1 (one non-zero singular value)
    FI_v = (W[1:] - W[:-1]) ** 2 / W[:-1]

    return np.sum(FI_v), {"Dimension": dimension, "Delay": delay, "Values": FI_v}
