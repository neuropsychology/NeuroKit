# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

from .fractal_correlation import fractal_correlation
from .utils_complexity_embedding import complexity_embedding


def complexity_dimension(signal, delay=1, dimension_max=20, method="afnn", show=False, **kwargs):
    """**Automated selection of the optimal Embedding Dimension (m)**

    The Embedding Dimension (*m*, sometimes referred to as *d* or *order*) is the second
    critical parameter (the first being the :func:`delay <complexity_delay>` :math:`\\tau`)
    involved in the construction of the time-delay embedding of a signal. It corresponds to the
    number of delayed states (versions of the signals lagged by :math:`\\tau`) that we include in
    the embedding.

    Though one can commonly find values of 2 or 3 used in practice, several authors suggested
    different numerical methods to guide the choice of *m*:

    * **Correlation Dimension** (CD): One of the earliest method to estimate the optimal *m*
      was to calculate the :func:`correlation dimension <fractal_correlation>` for embeddings of
      various sizes and look for a saturation (i.e., a plateau) in its value as the embedding
      dimension increases. One of the limitation is that a saturation will also occur when there is
      not enough data to adequately fill the high-dimensional space (note that, in general, having
      such large embeddings that it significantly shortens the length of the signal is not
      recommended).
    * **FNN** (False Nearest Neighbour): The method, introduced by Kennel et al. (1992), is based
      on the assumption that two points that are near to each other in the sufficient embedding
      dimension should remain close as the dimension increases. The algorithm checks the neighbours
      in increasing embedding dimensions until it finds only a negligible number of false
      neighbours when going from dimension :math:`m` to :math:`m+1`. This corresponds to the lowest
      embedding dimension, which is presumed to give an unfolded space-state reconstruction. This
      method can fail in noisy signals due to the futile attempt of unfolding the noise (and in
      purely random signals, the amount of false neighbors does not substantially drops as *m*
      increases). The **figure** below show how projections to higher-dimensional spaces can be
      used to detect false nearest neighbours. For instance, the red and the yellow points are
      neighbours in the 1D space, but not in the 2D space.

    .. figure:: ../img/douglas2022b.png
       :alt: Illustration of FNN (Douglas et al., 2022).

    * **AFN** (Average False Neighbors): This modification by Cao (1997) of the FNN method
      addresses one of its main drawback, the need for a heuristic choice for the tolerance
      thresholds *r*. It uses the maximal Euclidian distance to represent nearest neighbors, and
      averages all ratios of the distance in :math:`m+1` to :math:`m` dimension and defines *E1* and
      *E2* as parameters. The optimal dimension corresponds to when *E1* stops changing (reaches a
      plateau). E1 reaches a plateau at a dimension *d0* if the signal comes from an attractor.
      Then *d0*+1 is the optimal minimum embedding dimension. *E2* is a useful quantity to
      distinguish deterministic signals from stochastic signals. A constant *E2* close to 1 for any
      embedding dimension *d* suggests random data, since the future values are independent of the
      past values.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted Tau :math:`\\tau`, sometimes referred to as Lag) in samples.
        See :func:`complexity_delay()` to choose the optimal value for this parameter.
    dimension_max : int
        The maximum embedding dimension to test.
    method : str
        Can be ``"afn"`` (Average False Neighbor), ``"fnn"`` (False Nearest Neighbour), or ``"cd"``
        (Correlation Dimension).
    show : bool
        Visualize the result.
    **kwargs
        Other arguments, such as ``R=10.0`` or ``A=2.0`` (relative and absolute tolerance, only for
        ``'fnn'`` method).

    Returns
    -------
    dimension : int
        Optimal embedding dimension.
    parameters : dict
        A dictionary containing additional information regarding the parameters used
        to compute the optimal dimension.

    See Also
    ------------
    complexity, complexity_dimension, complexity_delay, complexity_tolerance

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=[5, 7, 8], noise=0.01)

      # Correlation Dimension
      @savefig p_complexity_dimension1.png scale=100%
      optimal_dimension, info = nk.complexity_dimension(signal,
                                                        delay=20,
                                                        dimension_max=10,
                                                        method='cd',
                                                        show=True)
      @suppress
      plt.close()

    .. ipython:: python

      # FNN
      @savefig p_complexity_dimension2.png scale=100%
      optimal_dimension, info = nk.complexity_dimension(signal,
                                                        delay=20,
                                                        dimension_max=20,
                                                        method='fnn',
                                                        show=True)
      @suppress
      plt.close()

    .. ipython:: python

      # AFNN
      @savefig p_complexity_dimension3.png scale=100%
      optimal_dimension, info = nk.complexity_dimension(signal,
                                                        delay=20,
                                                        dimension_max=20,
                                                        method='afnn',
                                                        show=True)
      @suppress
      plt.close()


    References
    -----------
    * Kennel, M. B., Brown, R., & Abarbanel, H. D. (1992). Determining embedding dimension for
      phase-space reconstruction using a geometrical construction. Physical review A, 45(6), 3403.
    * Cao, L. (1997). Practical method for determining the minimum embedding dimension of a scalar
      time series. Physica D: Nonlinear Phenomena, 110(1-2), 43-50.
    * Rhodes, C., & Morari, M. (1997). The false nearest neighbors algorithm: An overview.
      Computers & Chemical Engineering, 21, S1149-S1154.
    * Krakovská, A., Mezeiová, K., & Budáčová, H. (2015). Use of false nearest neighbours for
      selecting variables and embedding parameters for state space reconstruction. Journal of
      Complex Systems, 2015.
    * Gautama, T., Mandic, D. P., & Van Hulle, M. M. (2003, April). A differential entropy based
      method for determining the optimal embedding parameters of a signal. In 2003 IEEE
      International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings.
      (ICASSP'03). (Vol. 6, pp. VI-29). IEEE.

    """
    # Initialize vectors
    if isinstance(dimension_max, int):
        dimension_seq = np.arange(1, dimension_max + 1)
    else:
        dimension_seq = np.array(dimension_max)

    # Method
    method = method.lower()
    if method in ["afnn", "afn"]:
        # Append value (as it gets cropped afterwards anyway)
        dimension_seq = np.append(dimension_seq, [dimension_seq[-1] + 1])
        E, Es = _embedding_dimension_afn(signal, dimension_seq=dimension_seq, delay=delay, **kwargs)
        E1 = E[1:] / E[:-1]
        E2 = Es[1:] / Es[:-1]

        # To find where E1 saturates, set a threshold of difference
        # threshold = 0.1 * (np.max(E1) - np.min(E1))
        min_dimension = [i for i, x in enumerate(E1 >= 0.85 * np.nanmax(E1)) if x][0] + 1

        # To standardize the length of dimension_seq with E1 and E2
        dimension_seq = dimension_seq[:-1]

        # Store information
        info = {"Method": method, "Values": dimension_seq, "E1": E1, "E2": E2}

        if show is True:
            _embedding_dimension_plot(
                method=method,
                dimension_seq=dimension_seq,
                min_dimension=min_dimension,
                E1=E1,
                E2=E2,
            )

    elif method in ["fnn"]:
        f1, f2, f3 = _embedding_dimension_ffn(signal, dimension_seq=dimension_seq, delay=delay, **kwargs)

        min_dimension = [i for i, x in enumerate(f3 <= 1.85 * np.min(f3[np.nonzero(f3)])) if x][0]

        # Store information
        info = {"Method": method, "Values": dimension_seq, "f1": f1, "f2": f2, "f3": f3}

        if show is True:
            _embedding_dimension_plot(
                method=method,
                dimension_seq=dimension_seq,
                min_dimension=min_dimension,
                f1=f1,
                f2=f2,
                f3=f3,
            )

    elif method in ["correlation", "cd"]:
        CDs = _embedding_dimension_correlation(signal, dimension_seq, delay=delay, **kwargs)

        # Find elbow (TODO: replace by better method of elbow localization)
        min_dimension = dimension_seq[np.where(CDs >= 0.66 * np.max(CDs))[0][0]]

        # Store information
        info = {"Method": method, "Values": dimension_seq, "CD": CDs}

        if show is True:
            _embedding_dimension_plot(
                method=method,
                dimension_seq=dimension_seq,
                min_dimension=min_dimension,
                CD=CDs,
            )

    else:
        raise ValueError("NeuroKit error: complexity_dimension(): 'method' not recognized.")

    return min_dimension, info


# =============================================================================
# Methods
# =============================================================================
def _embedding_dimension_correlation(signal, dimension_seq, delay=1, **kwargs):
    """Return the Correlation Dimension (CD) for a all d in dimension_seq."""
    CDs = np.zeros(len(dimension_seq))
    for i, d in enumerate(dimension_seq):
        CDs[i] = fractal_correlation(signal, dimension=d, delay=delay, **kwargs)[0]

    return CDs


def _embedding_dimension_afn(signal, dimension_seq, delay=1, **kwargs):
    """AFN."""
    values = np.asarray(
        [_embedding_dimension_afn_d(signal, dimension, delay, **kwargs) for dimension in dimension_seq]
    ).T
    E, Es = values[0, :], values[1, :]

    return E, Es


def _embedding_dimension_afn_d(signal, dimension, delay=1, metric="chebyshev", window=10, maxnum=None, **kwargs):
    """Returns E(d) and E^*(d) for the AFN method for a single d.

    E(d) and E^*(d) will be used to calculate E1(d) and E2(d).
    E1(d) = E(d + 1)/E(d).
    E2(d) = E*(d + 1)/E*(d).

    """
    d, dist, index, y2 = _embedding_dimension_d(signal, dimension, delay, metric, window, maxnum)

    # Compute the ratio of near-neighbor distances in d + 1 over d dimension
    # Its average is E(d)
    if any(d == 0) or any(dist == 0):
        E = np.nan
        Es = np.nan
    else:
        E = np.mean(d / dist)
        # Calculate E^*(d)
        Es = np.mean(np.abs(y2[:, -1] - y2[index, -1]))

    return E, Es


def _embedding_dimension_ffn(signal, dimension_seq, delay=1, R=10.0, A=2.0, **kwargs):
    """Compute the fraction of false nearest neighbors.

    The false nearest neighbors (FNN) method described by Kennel et al.
    (1992) to calculate the minimum embedding dimension required to embed a scalar time series.

    Returns 3 vectors:

    - f1 : Fraction of neighbors classified as false by Test I.
    - f2 : Fraction of neighbors classified as false by Test II.
    - f3 : Fraction of neighbors classified as false by either Test I or Test II.

    """
    values = np.asarray(
        [_embedding_dimension_ffn_d(signal, dimension, delay, R=R, A=A, **kwargs) for dimension in dimension_seq]
    ).T
    f1, f2, f3 = values[0, :], values[1, :], values[2, :]

    return f1, f2, f3


def _embedding_dimension_ffn_d(signal, dimension, delay=1, R=10.0, A=2.0, metric="euclidean", window=10, maxnum=None):
    """Return fraction of false nearest neighbors for a single d."""
    d, dist, index, y2 = _embedding_dimension_d(signal, dimension, delay, metric, window, maxnum)

    # Find all potential false neighbors using Kennel et al.'s tests.
    dist[dist == 0] = np.nan  # assign nan to avoid divide by zero error in next line
    f1 = np.abs(y2[:, -1] - y2[index, -1]) / dist > R
    f2 = d / np.std(signal) > A
    f3 = f1 | f2

    return np.mean(f1), np.mean(f2), np.mean(f3)


# =============================================================================
# Internals
# =============================================================================
def _embedding_dimension_d(signal, dimension, delay=1, metric="chebyshev", window=10, maxnum=None):
    # We need to reduce the number of points in dimension d by tau
    # so that after reconstruction, there'll be equal number of points
    # at both dimension d as well as dimension d + 1.
    y1 = complexity_embedding(signal[:-delay], delay=delay, dimension=dimension)
    y2 = complexity_embedding(signal, delay=delay, dimension=dimension + 1)

    # Find near neighbors in dimension d.
    index, dist = _embedding_dimension_neighbors(y1, metric=metric, window=window, maxnum=maxnum)

    # Compute the near-neighbor distances in d + 1 dimension
    # TODO: is there a way to make this faster?
    d = [scipy.spatial.distance.chebyshev(i, j) for i, j in zip(y2, y2[index])]

    return np.asarray(d), dist, index, y2


def _embedding_dimension_neighbors(y, metric="chebyshev", window=0, maxnum=None, show=False):
    """Find nearest neighbors of all points in the given array. Finds the nearest neighbors of all points in the given
    array using SciPy's KDTree search.

    Parameters
    ----------
    y : ndarray
        embedded signal: N-dimensional array containing time-delayed vectors.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common
        to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003),
        or to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension_max : int
        The maximum embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order')
        to test.
    metric : str
        Metric to use for distance computation.  Must be one of "cityblock" (aka the Manhattan metric),
        "chebyshev" (aka the maximum norm metric), or "euclidean". Defaults to 'chebyshev'.
    window : int
        Minimum temporal separation (Theiler window) that should exist between near neighbors.
        This is crucial while computing Lyapunov exponents and the correlation dimension. Defaults to 0.
    maxnum : int
        Maximum number of near neighbors that should be found for each point.
        In rare cases, when there are no neighbors that are at a nonzero distance, this will have to
        be increased (i.e., beyond 2 * window + 3). Defaults to None (optimum).
    show : bool
        Defaults to False.

    Returns
    -------
    index : array
        Array containing indices of near neighbors.
    dist : array
        Array containing near neighbor distances.

    """
    if metric == "chebyshev":
        p = np.inf
    elif metric == "cityblock":
        p = 1
    elif metric == "euclidean":
        p = 2
    else:
        raise ValueError('Unknown metric. Should be one of "cityblock", ' '"euclidean", or "chebyshev".')

    tree = scipy.spatial.cKDTree(y)  # pylint: disable=E1102
    n = len(y)

    if not maxnum:
        maxnum = (window + 1) + 1 + (window + 1)
    else:
        maxnum = max(1, maxnum)

    if maxnum >= n:
        raise ValueError("maxnum is bigger than array length.")

    # Query for k numbers of nearest neighbors
    distances, indices = tree.query(y, k=range(1, maxnum + 2), p=p)

    # Substract the first point
    valid = indices - np.tile(np.arange(n), (indices.shape[1], 1)).T

    # Remove points that are closer than min temporal separation
    valid = np.abs(valid) > window

    # Remove also self reference (d > 0)
    valid = valid & (distances > 0)

    # Get indices to keep
    valid = (np.arange(len(distances)), np.argmax(valid, axis=1))

    distances = distances[valid]
    indices = indices[(valid)]

    if show is True:
        plt.plot(indices, distances)

    return indices, distances


# =============================================================================
# Plotting
# =============================================================================


def _embedding_dimension_plot(
    method,
    dimension_seq,
    min_dimension,
    E1=None,
    E2=None,
    f1=None,
    f2=None,
    f3=None,
    CD=None,
    ax=None,
):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    ax.set_title("Optimization of Dimension (d)")
    ax.set_xlabel("Embedding dimension $d$")

    if method in ["correlation", "cd"]:
        ax.set_ylabel("Correlation Dimension (CD)")
        ax.plot(dimension_seq, CD, "o-", label="$CD$", color="#852b01")
    else:
        ax.set_ylabel("$E_1(d)$ and $E_2(d)$")
        if method in ["afnn"]:
            ax.plot(dimension_seq, E1, "o-", label="$E_1(d)$", color="#FF5722")
            ax.plot(dimension_seq, E2, "o-", label="$E_2(d)$", color="#FFC107")

        if method in ["fnn"]:
            ax.plot(dimension_seq, 100 * f1, "o--", label="Test I", color="#FF5722")
            ax.plot(dimension_seq, 100 * f2, "^--", label="Test II", color="#f44336")
            ax.plot(dimension_seq, 100 * f3, "s-", label="Test I + II", color="#852b01")

    ax.axvline(x=min_dimension, color="#E91E63", label="Optimal dimension: " + str(min_dimension))
    ax.legend(loc="upper right")

    return fig
