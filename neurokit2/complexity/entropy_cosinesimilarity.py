import numpy as np
import pandas as pd

from ..stats import rescale
from .utils_complexity_embedding import complexity_embedding
from .optim_complexity_tolerance import complexity_tolerance


def entropy_cosinesimilarity(signal=None, delay=1, dimension=3, tolerance="sd"):
    """**Cosine Similarity Entropy (CoSiEn) and its multiscale variant (MSCoSiEn)**

    Cosine Similarity Entropy (CoSiEn) is based on fundamental modifications of the SampEn and the
    MSEn approaches, which makes the CoSiEn amplitude-independent and robust to spikes and short
    length of data segments, two key problems with the standard SampEn.

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
    **kwargs
        Optional arguments. Not used for now.

    Returns
    --------
    cosien : float
        The Cosine Similarity Entropy entropy of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_multiscale

    Examples
    ----------
    1. Compute Cosine Similarity Entropy (**CoSiEn**)
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=5)

      cosien, info = nk.entropy_cosinesimilarity(signal)
      cosien

    2. Compute Multiscale Cosine Similarity Entropy (**MSCoSiEn**)
    .. ipython:: python

      mscosien, info = nk.entropy_multiscale(signal, method="MSCoSiEn")
      mscosien

    References
    -----------
    * Chanwimalueang, T., & Mandic, D. P. (2017). Cosine similarity entropy: Self-correlation-based
      complexity analysis of dynamical systems. Entropy, 19(12), 652.

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

    # Steps from Chanwimalueang et al. (2017)
    # TODO: this should be integrated in utils.py - _get_count()
    # But so far I am not sure how to adapt this code to a logic similar to the one in _get_count()

    # 1. (Optional pre-processing) Remove the offset and generate a zero median series
    signal = signal - np.median(signal)

    # 2. Construct the embedding vectors
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)

    # 3. Compute angular distance for all pairwise embedding vectors
    norm = np.linalg.norm(embedded, axis=1)
    norm = np.triu(np.inner(embedded, embedded) / np.outer(norm, norm), 1)
    # Rescale to [-1, 1] to prevent numerical errors
    norm = rescale(norm, [-1, 1])
    # angular distance
    d = np.arccos(norm) / np.pi

    # 4. Obtain the number of similar patterns P(m)i(rCSE) when a criterion AngDis(m)i,j≤rCSE is
    # fulfilled.
    Pm = np.real(d) <= info["Tolerance"]

    # 5. Compute the local probability of occurrences of similar patterns
    # 6. Compute the global probability of occurrences of similar patterns
    N = len(signal) - (dimension - 1) * delay
    Bm = np.sum(np.triu(Pm, 1) / (N * (N - 1) / 2))

    # 7. Cosine similarity entropy
    if Bm == 1 or Bm == 0:
        cosien = 0
    else:
        cosien = -(Bm * np.log(Bm)) - ((1 - Bm) * np.log(1 - Bm))

    return cosien, info
