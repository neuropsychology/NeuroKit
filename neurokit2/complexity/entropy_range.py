from warnings import warn

import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from ..stats import standardize
from .entropy_approximate import entropy_approximate
from .optim_complexity_tolerance import complexity_tolerance
from .utils_complexity_embedding import complexity_embedding
from .utils_entropy import _phi_divide


def entropy_range(signal, dimension=3, delay=1, tolerance="sd", approximate=False, corrected=False):
    """**Range Entropy (RangEn)**

    Introduced by Omidvarnia et al. (2018), Range Entropy (RangEn or RangeEn) refers to a modified
    form of SampEn (or ApEn).

    Both ApEn and SampEn compute the logarithmic likelihood that runs of patterns that are close
    remain close on the next incremental comparisons, of which this closeness is estimated by the
    Chebyshev distance. Range Entropy uses instead a normalized "range distance", resulting in
    modified forms of ApEn and SampEn, **RangEn (A)** (*mApEn*) and **RangEn (B)** (*mSampEn*).

    The range distance between two embedded vectors :math:`X_i` and :math:`X_j` is

    .. math::

      d_{range}(X_i, X_j) = \\frac{\\max_k |x_{i+k} - x_{j+k}| - \\min_k |x_{i+k} - x_{j+k}|}
      {\\max_k |x_{i+k} - x_{j+k}| + \\min_k |x_{i+k} - x_{j+k}|}

    which lies between 0 and 1 and does not depend on the amplitude of the signal. It is undefined
    (0 / 0) for identical vectors, which are therefore never counted as matches, and neither is a
    vector with itself. Two vectors match when :math:`d_{range} \\leq r`. **RangEn (B)** is then
    :func:`SampEn <entropy_sample>` with this distance, i.e., :math:`-\\ln(A / B)`, where *B* and *A*
    are the proportions of the pairs of the :math:`N - m \\tau` templates that match at dimensions
    *m* and *m* + 1. **RangEn (A)** is :func:`ApEn <entropy_approximate>` with this distance,
    :math:`\\Phi^m(r) - \\Phi^{m+1}(r)`, where :math:`\\Phi^m(r)` is the average of the logarithm
    of the proportion of the vectors that match each vector.

    However, as the vectors do not match themselves, RangEn (A), unlike ApEn, is undefined (*NaN*
    or *Inf*) as soon as one vector has no match, which often happens. As such, using RangEn (B)
    is recommended instead.

    RangEn is described as more robust to nonstationary signal changes, and has a more linear
    relationship with the Hurst exponent (compared to ApEn and SampEn), and has no need for signal
    amplitude correction.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    tolerance : float
        Tolerance (often denoted as *r*), the range distance (between 0 and 1) under which two
        vectors are considered similar. The range distance being normalized, the tolerance is not
        multiplied by the standard deviation (SD) of the signal (Omidvarnia et al., 2018). If
        ``"sd"`` (default), will be set to 0.2 (i.e., :math:`0.2 * SD` of the standardized
        signal). Other methods of :func:`complexity_tolerance` are also applied to the
        standardized signal.
    approximate : bool
        The entropy algorithm to use. If ``False`` (default), will use sample entropy and return
        *mSampEn* (**RangEn B**). If ``True``, will use approximate entropy and return *mApEn*
        (**RangEn A**).
    corrected : bool
        If ``True`` (only with ``approximate=True``), will compute the :func:`corrected
        <entropy_approximate>` version of ApEn (cApEn) with the range distance.

    See Also
    --------
    entropy_approximate, entropy_sample

    Returns
    -------
    RangEn : float
        Range Entropy. If undefined conditional probabilities are detected (logarithm
        of sum of conditional probabilities is ``ln(0)``), ``np.inf`` will
        be returned, meaning it fails to retrieve 'accurate' regularity information.
        This tends to happen for short data segments, increasing tolerance
        levels might help avoid this.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, sampling_rate=100, frequency=[5, 6])

      # Range Entropy B (mSampEn)
      RangEnB, info = nk.entropy_range(signal, approximate=False)
      RangEnB

      # Range Entropy A (mApEn)
      RangEnA, info = nk.entropy_range(signal, approximate=True)
      RangEnA

      # Range Entropy A (corrected)
      RangEnAc, info = nk.entropy_range(signal, approximate=True, corrected=True)
      RangEnAc

    References
    ----------
    * Omidvarnia, A., Mesbah, M., Pedersen, M., & Jackson, G. (2018). Range entropy: A bridge
      between signal complexity and self-similarity. Entropy, 20(12), 962.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    if corrected is True and approximate is False:
        raise ValueError("The corrected version of RangEn is only available for RangEn (A) (`approximate=True`).")
    signal = np.asarray(signal, dtype=float)

    # Store parameters
    info = {"Dimension": dimension, "Delay": delay}

    # The range distance is undefined between identical vectors (e.g., of a constant signal)
    if np.any(np.isnan(signal)) or np.all(signal == signal[0]):
        warn(
            "The signal is constant or contains missing values (NaN): RangEn is returned as NaN.",
            category=NeuroKitWarning,
        )
        info["Tolerance"] = np.nan
        return np.nan, info

    info["Tolerance"] = _entropy_range_tolerance(signal, tolerance, delay, dimension)

    if corrected is True:  # cApEn with the range distance
        return entropy_approximate(
            signal,
            delay=delay,
            dimension=dimension,
            tolerance=info["Tolerance"],
            corrected=True,
            distance="range",
        )[0], info

    # The vectors at dimension m + 1, and those at m (``delay`` more)
    embedded2 = complexity_embedding(signal, delay=delay, dimension=dimension + 1)
    embedded1 = complexity_embedding(signal, delay=delay, dimension=dimension)

    with np.errstate(divide="ignore", invalid="ignore"):
        if approximate is False:  # mSampEn - RangEn (B)
            # Proportion of the pairs of the same N - m * delay templates that match at m and m + 1
            # (the counts exclude the vector itself)
            n = len(embedded2)
            phi = [np.sum(_entropy_range_count(e[:n], info["Tolerance"])) / (n * (n - 1)) for e in [embedded1, embedded2]]
            return _phi_divide(phi), info

        # mApEn - RangEn (A): average log proportion of matches (the vector itself not being one)
        phi = [np.mean(np.log(_entropy_range_count(e, info["Tolerance"]) / len(e))) for e in [embedded1, embedded2]]
        return phi[0] - phi[1], info


# =============================================================================
# Utils
# =============================================================================
def _entropy_range_tolerance(signal, tolerance, delay, dimension):
    """Tolerance for normalized distances (between 0 and 1, e.g., the range or angular distance),
    which do not depend on the amplitude of the signal: the methods of complexity_tolerance() are
    applied to the standardized signal ("sd" gives 0.2)."""
    if not isinstance(tolerance, str):
        return tolerance
    if np.std(signal) == 0:  # Constant signal (cannot be standardized)
        return complexity_tolerance(signal, method=tolerance, delay=delay, dimension=dimension, show=False)[0]
    r = complexity_tolerance(standardize(signal), method=tolerance, delay=delay, dimension=dimension, show=False)[0]
    # Remove the rounding error of the standardization (e.g., 0.20000000000000004 for "sd")
    return np.round(r, 10)


def _entropy_range_count(embedded, tolerance):
    """Number of the other vectors within the tolerance (range distance) of each vector. The range
    distance is undefined (0 / 0) for identical vectors, which are thus never counted (neither is
    the vector itself)."""
    n, m = embedded.shape
    count = np.zeros(n)
    # Process blocks of rows (of about 65,000 pairs, which fit in the cache) to limit the memory used
    block = max(1, 2**16 // n)
    with np.errstate(divide="ignore", invalid="ignore"):
        for start in range(0, n, block):
            rows = embedded[start : start + block]
            # Largest and smallest absolute differences between the elements of the vectors
            dmax = np.abs(rows[:, None, 0] - embedded[None, :, 0])
            dmin = dmax.copy()
            for k in range(1, m):
                diff = np.abs(rows[:, None, k] - embedded[None, :, k])
                np.maximum(dmax, diff, out=dmax)
                np.minimum(dmin, diff, out=dmin)
            # NaN (not counted) for identical vectors
            count[start : start + block] = np.sum((dmax - dmin) / (dmax + dmin) <= tolerance, axis=1)
    return count
