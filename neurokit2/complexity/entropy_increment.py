import numpy as np
import pandas as pd

from .utils_complexity_embedding import complexity_embedding
from .entropy_shannon import entropy_shannon


def entropy_increment(signal, dimension=2, q=4, **kwargs):
    """**Increment Entropy (IncrEn) and its Multiscale variant (MSIncrEn)**

    Increment Entropy (IncrEn) quantifies the magnitudes of the variations between adjacent
    elements into ranks based on a precision factor *q* and the standard deviation of the time
    series. IncrEn is conceptually similar to :func:`permutation entropy <entropy_permutation>` in
    that it also uses the concepts of symbolic dynamics.

    In the IncrEn calculation, two letters are used to describe the relationship between adjacent
    elements in a time series. One letter represents the volatility direction, and the other
    represents the magnitude of the variation between the adjacent elements.

    The time series is reconstructed into vectors of *m* elements. Each element of each vector
    represents the increment between two neighbouring elements in the original time series.
    Each increment element is mapped to a word consisting of two letters (one letter represents
    the volatility direction, and the other represents the magnitude of the variation between
    the adjacent elements), and then, each vector is described as a symbolic (discrete) pattern.
    The :func:`Shannon entropy <entropy_shannon>` of the probabilities of independent patterns is
    then computed.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    q : float
        The quantifying resolution *q* represents the precision of *IncrEn*, with larger values
        indicating a higher precision, causing IncrEn to be more sensitive to subtle fluctuations.
        The IncrEn value increases with increasing *q*, until reaching a plateau. This property can
        be useful to selecting an optimal *q* value.
    **kwargs : optional
        Other keyword arguments, such as the logarithmic ``base`` to use for
        :func:`entropy_shannon`.

    Returns
    --------
    incren : float
        The Increment Entropy of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used, such as the
        average entropy ``AvEn``.

    See Also
    --------
    entropy_shannon, entropy_multiscale

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal
      signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)

      # IncrEn
      incren, _ = nk.entropy_increment(signal, dimension=3, q=2)
      incren

      # Multiscale IncrEn (MSIncrEn)
      @savefig p_entropy_increment1.png scale=100%
      msincren, _ = nk.entropy_multiscale(signal, method="MSIncrEn", show=True)
      @suppress
      plt.close()

    References
    -----------
    * Liu, X., Jiang, A., Xu, N., & Xue, J. (2016). Increment entropy as a measure of complexity
      for time series. Entropy, 18(1), 22.
    * Liu, X., Jiang, A., Xu, N., & Xue, J. (2016). Correction on Liu, X.; Jiang, A.; Xu, N.; Xue,
      J. Increment Entropy as a Measure of Complexity for Time Series. Entropy 2016, 18, 22.
      Entropy, 18(4), 133.
    * Liu, X., Wang, X., Zhou, X., & Jiang, A. (2018). Appropriate use of the increment entropy for
      electrophysiological time series. Computers in Biology and Medicine, 95, 13-23.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Store parameters
    info = {"Dimension": dimension, "q": q}

    # Time-embedding of the consecutive differences ("increment series")
    embedded = complexity_embedding(np.diff(signal), dimension=dimension, **kwargs)

    # The sign indicates the direction of the volatility between the corresponding neighbouring
    # elements in the original time series; it takes values of 1, 0, or 1, indicating a rise, no
    # change, or a decline
    sign = np.sign(embedded)

    # The size describes the magnitude of the variation between these adjacent elements
    Temp = np.tile(np.std(embedded, axis=1, ddof=1, keepdims=True), (1, dimension))
    size = np.minimum(q, np.floor(abs(embedded) * q / Temp))
    size[np.any(Temp == 0, axis=1), :] = 0

    # Each element in each vector is mapped to a word consisting of the sign and the size
    words = sign * size

    # Get probabilities of occurence
    freq = np.unique(words, axis=0)
    freq = [np.sum(~np.any(words - freq[k, :], axis=1)) for k in range(len(freq))]
    freq = np.array(freq) / np.sum(freq)

    # Compute entropy
    incren, _ = entropy_shannon(freq=freq, **kwargs)

    # Normalize
    incren = incren / (dimension - 1)

    return incren, info
