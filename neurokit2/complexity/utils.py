# -*- coding: utf-8 -*-
import numpy as np
import sklearn.neighbors

from .embedding import embedding





# =============================================================================
# Phi
# =============================================================================


def _phi(signal, delay=1, dimension=2, r="default", distance='chebyshev', approximate=True, fuzzy=False):
    """Common internal for `entropy_approximate` and `entropy_sample`.

    Adapted from `EntroPy <https://github.com/raphaelvallat/entropy>`_, check it out!
    """
    # Initialize phi
    phi = np.zeros(2)

    embedded1, count1 = _get_embedded(signal, delay, dimension, r, distance=distance, approximate=approximate, fuzzy=fuzzy)
    embedded2, count2 = _get_embedded(signal, delay, dimension + 1, r, distance=distance, approximate=True, fuzzy=fuzzy)

    if approximate is True:
        phi[0] = np.mean(np.log(count1 / embedded1.shape[0]))
        phi[1] = np.mean(np.log(count2 / embedded2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (embedded1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (embedded2.shape[0] - 1))
    return phi




def _phi_divide(phi):
    if phi[0] == 0:
        return -np.inf
    division = np.divide(phi[1], phi[0])
    if division == 0:
        return np.inf
    return -np.log(division)




# =============================================================================
# Get Embedded
# =============================================================================


def _get_embedded(signal, delay=1, dimension=2, r="default", distance='chebyshev', approximate=True, fuzzy=False):
    """
    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>> delay = nk.embedding_delay(signal)
    >>>
    >>> embbeded, count = _get_embedded(signal,
                                        delay,
                                        r=0.2 * np.std(signal, ddof=1),
                                        dimension=2,
                                        metric='chebyshev',
                                        approximate=False)
    """
    # Sanity checks
    if distance not in sklearn.neighbors.KDTree.valid_metrics:
        raise ValueError("NeuroKit error: _get_embedded(): The given metric (%s) is not valid. The valid metric names are: %s" % (distance, sklearn.neighbors.KDTree.valid_metrics))

    # Get embedded
    embedded = embedding(signal, delay=delay, dimension=dimension)
    if approximate is False:
        embedded = embedded[:-1]  # Removes the last line

    if fuzzy is False:
        # Get neighbors count
        count = _get_count(embedded, r=r, distance=distance)
    else:
        # FuzzyEn: Remove the local baselines of vectors
        embedded -= np.mean(embedded, axis=1, keepdims=True)
        count = _get_count_fuzzy(embedded, r=r, distance=distance, n=1)


    return embedded, count


# =============================================================================
# Get Count
# =============================================================================
def _get_count(embedded, r, distance="chebyshev"):
    """
    """
    kdtree = sklearn.neighbors.KDTree(embedded, metric=distance)
    count = kdtree.query_radius(embedded, r, count_only=True).astype(np.float64)
    return count


def _get_count_fuzzy(embedded, r, distance="chebyshev", n=1):
    dist = sklearn.neighbors.DistanceMetric.get_metric(distance)
    dist = dist.pairwise(embedded)

    # TODO: can this be optimized?
    sim = np.exp(-np.linalg.matrix_power(dist, n) / r)
    count = np.sum(sim, axis=0)
    return count


# =============================================================================
# Get R
# =============================================================================
def _get_r(signal, r="default"):

    if isinstance(r, str) or (r is None):
        r = 0.2 * np.std(signal, ddof=1)

    return r


# =============================================================================
# Get Scale Factor
# =============================================================================
def _get_scale(signal, scale="default", dimension=2):
    # Select scale
    if scale is None or scale == "max":
        scale = np.arange(1, len(signal) // 2)  # Set to max
    elif scale == "default":
        scale = np.arange(1, int(len(signal) / (dimension + 10)))  # See https://github.com/neuropsychology/NeuroKit/issues/75#issuecomment-583884426
    elif isinstance(scale, int):
        scale = np.range(1, scale)

    return scale

# =============================================================================
# Get Coarsegrained
# =============================================================================
def _get_coarsegrained_rolling(signal, scale=2):
    """Used in composite multiscale entropy.
    """
    if scale in [0, 1]:
        return np.array([signal])
    if scale > len(signal):
        return np.array([])

    n = len(signal)
    j_max = n // scale
    k_max = scale

    if n < 2:
        raise ValueError("NeuroKit error: _get_coarsegrained_rolling(): The signal is too short!")

    coarsed = np.full([k_max, j_max], np.nan)
    for k in range(k_max):
        y = _get_coarsegrained(signal[k::], scale=scale, force=True)[0:j_max]
        coarsed[k, :] = y
    return coarsed


def _get_coarsegrained(signal, scale=2, force=False):
    """Extract coarse-grained time series.

    The coarse-grained time series for a scale factor Tau are obtained by
    calculating the arithmetic mean of Tau neighboring values without overlapping.

    To obtain the coarse-grained time series at a scale factor of Tau ,the original
    time series is divided into non-overlapping windows of length Tau and the
    data points inside each window are averaged.

    This coarse-graining procedure is similar to moving averaging and the decimation of the original time series.
    The decimation procedure shortens the length of the coarse-grained time series by a factor of Tau.

    This is an efficient version of ``pd.Series(signal).rolling(window=scale).mean().iloc[0::].values[scale-1::scale]``.
    >>> import neurokit2 as nk
    >>> signal = [0, 2, 4, 6, 8, 10]
    >>> cs = _get_coarsegrained(signal, scale=2)
    """
    if scale in [0, 1]:
        return signal
    n = len(signal)
    if force is True:
        # Get max j
        j = int(np.ceil(n / scale))
        # Extend signal by repeating the last element so that it matches the theorethical length
        signal = np.concatenate([signal, np.repeat(signal[-1], (j * scale) - len(signal))])
    else:
        j = n // scale
    x = np.reshape(signal[0:j*scale], (j, scale))
    coarsed = np.mean(x, axis=1)
    return coarsed

# =============================================================================
# ADDED
# =============================================================================
def _get_coarsegrained_rolling_tam(signal, scale=2):

    if scale in [0, 1]:
        return signal
    N = len(signal)
    j_max = N // scale
    k_max = scale

    k_coarsed = []

    for k in range(1, k_max + 1):
        k_coarsed_list = []
        for j in range(1, j_max + 1):
            start = (j-1) * scale + k
            end = j * scale + k - 1
            group_mean = np.mean(signal[start - 1:end])
            k_coarsed_list.append(group_mean)
        k_coarsed.append(k_coarsed_list)

    return np.array(k_coarsed)
