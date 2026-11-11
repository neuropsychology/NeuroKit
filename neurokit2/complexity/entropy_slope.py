from warnings import warn

import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from .entropy_shannon import entropy_shannon
from .utils_complexity_embedding import complexity_embedding


def entropy_slope(signal, dimension=3, thresholds=[0.1, 45], **kwargs):
    """**Slope Entropy (SlopEn)**

    Slope Entropy (SlopEn) uses an alphabet of three symbols, 0, 1, and 2, with positive (+) and
    negative versions (-) of the last two. Each symbol covers a range of slopes for the segment
    joining two consecutive samples of the input data, and the :func:`Shannon entropy <entropy_shannon>`
    of the relative frequency of each pattern is computed.

    Each vector of *m* (``dimension``) successive samples is transformed into the *m - 1* angles
    (in degrees, between -90 and 90) of the slopes between its consecutive samples, which are
    mapped to symbols with the angular ``thresholds``. With the default two thresholds (in
    Cuesta-Frau's notation, :math:`\\delta` and :math:`\\gamma`, here as angles), an angle *a* is
    mapped to 0 if :math:`|a| \\leq \\delta` (a tie), to +1 (or -1) if
    :math:`\\delta < |a| \\leq \\gamma`, and to +2 (or -2) if :math:`|a| > \\gamma`, depending on the
    sign of the slope. More thresholds add more symbols in the same way (an angle above the *q*-th
    threshold and up to the next one is mapped to :math:`\\pm q`). SlopEn is the entropy of the
    relative frequencies of the patterns of *m - 1* symbols.

    .. figure:: ../img/cuestafrau2019.png
       :alt: Figure from Cuesta-Frau, D. (2019).
       :target: https://doi.org/10.3390/e21121167

    Note that Cuesta-Frau (2019) divides the number of occurrences of each pattern by the number
    of different patterns found, rather than by the number of vectors, so that the "probabilities"
    do not sum to 1 (e.g., for the worked example of the paper, 5.29 rather than the 3.54 bits
    returned here). As in the paper, the signal is not standardized: the slopes, and therefore
    SlopEn, depend on its amplitude.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*), at least 2. See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    thresholds : list
        Angular thresholds (called *levels*). A list of monotonically increasing values in the
        range [0, 90] degrees. The default, ``[0.1, 45]``, corresponds to differences between
        consecutive samples of :math:`\\delta = \\tan(0.1^\\circ) \\approx 0.0017` and
        :math:`\\gamma = \\tan(45^\\circ) = 1` (Cuesta-Frau, 2019, used :math:`\\delta = 0.001`,
        i.e., an angle of 0.0573 degrees).
    **kwargs : optional
        Other keyword arguments, such as the ``delay`` between the samples of the vectors (1 by
        default, as in Cuesta-Frau, 2019), or the logarithmic ``base`` to use for
        :func:`entropy_shannon`.

    Returns
    -------
    slopen : float
        Slope Entropy of the signal. It is NaN (with a warning) if the signal contains missing
        values.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_shannon

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal
      signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)

      # Compute Slope Entropy
      slopen, info = nk.entropy_slope(signal, dimension=3, thresholds=[0.1, 45])
      slopen

      slopen, info = nk.entropy_slope(signal, dimension=3, thresholds=[5, 45, 60, 90])
      slopen

      # Compute Multiscale Slope Entropy (MSSlopEn)
      @savefig p_entropy_slope1.png scale=100%
      msslopen, info = nk.entropy_multiscale(signal, method="MSSlopEn", show=True)
      @suppress
      plt.close()

    References
    ----------
    * Cuesta-Frau, D. (2019). Slope entropy: A new time series complexity estimator based on both
      symbolic patterns and amplitude information. Entropy, 21(12), 1167.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    if dimension < 2:
        raise ValueError("The dimension of Slope Entropy should be at least 2.")
    levels = np.asarray(thresholds, dtype=float)
    if levels.ndim != 1 or len(levels) == 0 or np.any(np.diff(levels) <= 0) or levels[0] < 0 or levels[-1] > 90:
        raise ValueError("`thresholds` should be a list of increasing angles between 0 and 90 degrees.")

    # We could technically expose the Delay, but the paper is about consecutive differences so...
    delay = kwargs.pop("delay", 1)

    # Store parameters
    info = {"Dimension": dimension, "Delay": delay, "Thresholds": thresholds}

    signal = np.asarray(signal, dtype=float)
    if np.any(np.isnan(signal)):
        warn(
            "The signal contains missing values, whose slopes are not defined. Returning NaN.",
            category=NeuroKitWarning,
        )
        return np.nan, info

    # each subsequence of length m drawn from x (with a delay between its samples, 1 by default)
    # can be transformed into another subsequence of length m-1 with the differences of each pair
    # of consecutive samples, expressed as angles
    embedded = complexity_embedding(signal, dimension=dimension, delay=delay)
    angles = np.degrees(np.arctan(np.diff(embedded, axis=1)))

    # a threshold or thresholds must be applied to these differences in order to find the
    # corresponding symbolic representation: the number of thresholds below the absolute angle
    # (0 up to the first one included, q above the q-th threshold and up to the next one included),
    # with the sign of the slope
    symbols = (np.sign(angles) * np.searchsorted(levels, np.abs(angles), side="left")).astype(int)

    _, freq = np.unique(symbols, axis=0, return_counts=True)

    # Shannon Entropy
    slopen, _ = entropy_shannon(freq=freq / freq.sum(), **kwargs)

    return slopen, info
