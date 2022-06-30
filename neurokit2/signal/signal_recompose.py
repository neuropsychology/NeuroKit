import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster

from .signal_zerocrossings import signal_zerocrossings


def signal_recompose(components, method="wcorr", threshold=0.5, keep_sd=None, **kwargs):
    """**Combine signal sources after decomposition**

    Combine and reconstruct meaningful signal sources after signal decomposition.

    Parameters
    -----------
    components : array
        Array of components obtained via :func:`.signal_decompose`.
    method : str
        The decomposition method. Can be one of ``"wcorr"``.
    threshold : float
        The threshold used to group components together.
    keep_sd : float
        If a float is specified, will only keep the reconstructed components that are superior
        or equal to that percentage of the max standard deviaiton (SD) of the components. For
        instance, ``keep_sd=0.01`` will remove all components with SD lower than 1% of the
        max SD. This can be used to filter out noise.
    **kwargs
        Other arguments used to override, for instance ``metric="chebyshev"``.

    Returns
    -------
    Array
        Components of the recomposed components.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Create complex signal
      signal = nk.signal_simulate(duration=10, frequency=1, noise=0.01)  # High freq
      signal += 3 * nk.signal_simulate(duration=10, frequency=3, noise=0.01)  # Higher freq
      signal += 3 * np.linspace(0, 2, len(signal))  # Add baseline and trend
      signal += 2 * nk.signal_simulate(duration=10, frequency=0.1, noise=0)

      # Decompose signal
      components = nk.signal_decompose(signal, method='emd')

      # Recompose
      recomposed = nk.signal_recompose(components, method='wcorr', threshold=0.90)
      @savefig p_signal_recompose1.png scale=100%
      nk.signal_plot(components)  # Visualize components
      @suppress
      plt.close()

    """
    # Apply method
    method = method.lower()
    if method in ["wcorr"]:
        clusters = _signal_recompose_wcorr(components, threshold=threshold, **kwargs)
        recomposed = _signal_recompose_sum(components, clusters)
    else:
        raise ValueError("NeuroKit error: signal_decompose(): 'method' should be one of 'emd'")

    if keep_sd is not None:
        recomposed = _signal_recompose_filter_sd(components, threshold=keep_sd)

    return recomposed


# =============================================================================
# Recombination methods
# =============================================================================
def _signal_recompose_sum(components, clusters):
    # Reorient components
    components = components.T

    # Reconstruct Time Series from correlated components
    clusters = [np.where(clusters == cluster)[0] for cluster in np.unique(clusters)]

    if len(clusters) == 0:
        raise ValueError(
            "Not enough clusters of components detected. Please decrease the " "`threshold`."
        )
    # Initialize components matrix
    recomposed = np.zeros((len(components), len(clusters)))
    for i, indices in enumerate(clusters):
        recomposed[:, i] = components[:, indices].sum(axis=1)
    return recomposed.T


# =============================================================================
# Clustering Methods
# =============================================================================

# Weighted Correlation
# ----------------------------------------------------------------------------
def _signal_recompose_wcorr(components, threshold=0.5, metric="chebyshev"):
    """"""
    # Calculate the w-correlation matrix.
    wcorr = _signal_recompose_get_wcorr(components, show=False)

    # Find clusters in correlation matrix
    pairwise_distances = scipy.cluster.hierarchy.distance.pdist(wcorr, metric=metric)
    linkage = scipy.cluster.hierarchy.linkage(pairwise_distances, method="complete")
    threshold = threshold * pairwise_distances.max()
    clusters = scipy.cluster.hierarchy.fcluster(linkage, threshold, "distance")

    return clusters


def _signal_recompose_get_wcorr(components, show=False):
    """Calculates the weighted correlation matrix for the time series.

    References
    ----------
    - https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition

    """
    # Reorient components
    components = components.T

    L = components.shape[1]
    K = components.shape[0] - L + 1

    # Calculate the weights
    w = np.array(list(np.arange(L) + 1) + [L] * (K - L - 1) + list(np.arange(L) + 1)[::-1])

    def w_inner(F_i, F_j):
        return w.dot(F_i * F_j)

    # Calculated weighted norms, ||F_i||_w, then invert.
    F_wnorms = np.array([w_inner(components[:, i], components[:, i]) for i in range(L)])
    F_wnorms = F_wnorms ** -0.5

    # Calculate Wcorr.
    Wcorr = np.identity(L)
    for i in range(L):
        for j in range(i + 1, L):
            Wcorr[i, j] = abs(
                w_inner(components[:, i], components[:, j]) * F_wnorms[i] * F_wnorms[j]
            )
            Wcorr[j, i] = Wcorr[i, j]

    if show is True:
        ax = plt.imshow(Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0, 1)

        # For plotting purposes:
        min_range = 0
        max_range = len(Wcorr) - 1

        plt.xlim(min_range - 0.5, max_range + 0.5)
        plt.ylim(max_range + 0.5, min_range - 0.5)

    return Wcorr


# =============================================================================
# Filter method
# =============================================================================
def _signal_recompose_filter_sd(components, threshold=0.01):
    """Filter by standard deviation."""
    SDs = [np.std(components[i, :], ddof=1) for i in range(len(components))]
    indices = np.where(SDs >= threshold * np.max(SDs))
    return components[indices]


def _signal_recompose_meanfreq(components, sampling_rate=1000):
    """Get the mean frequency of components."""
    duration = components.shape[1] / sampling_rate
    n = len(components)
    freqs = np.zeros(n)

    for i in range(n):
        c = components[i, :] - np.mean(components[i, :])
        freqs[i] = len(signal_zerocrossings(c)) / duration
