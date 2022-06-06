import numpy as np
import pandas as pd

from .entropy_shannon import entropy_shannon


def entropy_slope(signal, dimension=3, thresholds=[0.1, 45], **kwargs):
    """**Slope Entropy (SlopEn)**

    Slope Entropy (SlopEn) uses an alphabet of three symbols, 0, 1, and 2, with positive (+) and
    negative versions (-) of the last two. Each symbol covers a range of slopes for the segment
    joining two consecutive samples of the input data, and the :func:`Shannon entropy <entropy_shannon>`
    of the relative frequency of each pattern is computed.

    .. figure:: ../img/cuestafrau2019.png
       :alt: Figure from Cuesta-Frau, D. (2019).
       :target: https://doi.org/10.3390/e21121167

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    thresholds : list
        Angular thresholds (called *levels*). A list of monotonically increasing  values in the
        range [0, 90] degrees.
    **kwargs : optional
        Other keyword arguments, such as the logarithmic ``base`` to use for
        :func:`entropy_shannon`.

    Returns
    -------
    slopen : float
        Slope Entropy of the signal.
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
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Store parameters
    info = {"Dimension": dimension}

    # We could technically expose the Delay, but the paper is about consecutive differences so...
    if "delay" in kwargs.keys():
        delay = kwargs["delay"]
        kwargs.pop("delay")
    else:
        delay = 1

    # each subsequence of length m drawn from x, can be transformed into another subsequence of
    # length m-1 with the differences of each pair of consecutive samples
    Tx = np.degrees(np.arctan(signal[delay:] - signal[:-delay]))
    N = len(Tx)

    # a threshold or thresholds must be applied to these differences in order to find the
    # corresponding symbolic representation
    symbols = np.zeros(N)
    for q in range(1, len(thresholds)):
        symbols[np.logical_and(Tx <= thresholds[q], Tx > thresholds[q - 1])] = q
        symbols[np.logical_and(Tx >= -thresholds[q], Tx < -thresholds[q - 1])] = -q

        if q == len(thresholds) - 1:
            symbols[Tx > thresholds[q]] = q + 1
            symbols[Tx < -thresholds[q]] = -(q + 1)

    unique = np.array([symbols[k : N - dimension + k + 1] for k in range(dimension - 1)]).T
    _, freq = np.unique(unique, axis=0, return_counts=True)

    # Shannon Entropy
    slopen, _ = entropy_shannon(freq=freq / freq.sum(), **kwargs)

    return slopen, info
