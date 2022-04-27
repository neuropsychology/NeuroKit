import matplotlib.cm
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .entropy_sample import entropy_sample


def entropy_hierarchical(signal, scale=3, show=False, **kwargs):
    """**Hierarchical Entropy (HEn)**

    Hierarchical Entropy (HEn) can be viewed as a generalization of the multiscale
    decomposition used in :func:`multiscale entropy <entropy_multiscale>`, and the Haar wavelet
    decomposition since it generate subtrees of the hierarchical tree. It preserves the strength of
    the multiscale decomposition with additional components of higher frequency in different
    scales. The hierarchical decomposition, unlike the wavelet decomposition, contains redundant
    components, which makes it sensitive to the dynamical richness of the time series.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    scale : int
        The number of scale factors.
    method : str
        Method for symbolic sequence partitioning. Can be one of ``"MEP"`` (default),
        ``"linear"``, ``"uniform"``, ``"kmeans"``.
    **kwargs : optional
        Other keyword arguments (currently not used).

    Returns
    -------
    SyDyEn : float
        Symbolic Dynamic Entropy (SyDyEn) of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_shannon, entropy_multiscale

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal
      signal = nk.signal_simulate(duration=20, sampling_rate=200, frequency=[5, 6], noise=0.5)

      # Compute Hierarchical Entropy (HEn)
      @savefig p_entropy_hierarchical1.png scale=100%
      hen, info = nk.entropy_hierarchical(signal, scale=5, show=True)
      @suppress
      plt.close()

    References
    ----------
    * Jiang, Y., Peng, C. K., & Xu, Y. (2011). Hierarchical entropy analysis for biological
      signals. Journal of Computational and Applied Mathematics, 236(5), 728-742.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Store parameters
    info = {}

    # TODO: Simplify this code, make it clearer and step by step, following the paper more closely

    Q, N = _hierarchical_decomposition(signal, scale=scale)

    HEns = np.zeros(len(Q))
    for T in range(len(Q)):
        Temp = Q[T, : int(N / (2 ** (int(np.log2(T + 1)))))]
        HEns[T], _ = entropy_sample(Temp, delay=1)

    Sn = np.zeros(scale)
    for t in range(scale):
        vals = HEns[(2 ** t) - 1 : (2 ** (t + 1)) - 1]
        Sn[t] = np.mean(vals[np.isfinite(vals)])

    # The HEn index is quantified as the area under the curve (AUC),
    # which is like the sum normalized by the number of values. It's similar to the mean.
    hen = np.trapz(Sn[np.isfinite(Sn)]) / len(Sn[np.isfinite(Sn)])

    if show is True:

        # Color normalization values by extending beyond the range of the mean values
        colormin = np.min(Sn) - np.ptp(Sn) * 0.1
        colormax = np.max(Sn) + np.ptp(Sn) * 0.1

        plt.figure()
        G = matplotlib.gridspec.GridSpec(10, 1)
        ax1 = plt.subplot(G[:2, :])
        ax1.plot(np.arange(1, scale + 1), Sn, color="black", zorder=0)
        ax1.scatter(
            np.arange(1, scale + 1),
            Sn,
            c=Sn,
            zorder=1,
            # Color map and color normalization values
            cmap="spring",
            vmin=colormin,
            vmax=colormax,
        )
        ax1.set_xticks(np.arange(1, scale + 1))
        ax1.set_xlabel("Scale Factor")
        ax1.set_ylabel("Entropy")
        ax1.set_title("Hierarchical Entropy")

        N = 2 ** (scale - 1)
        x = np.zeros(2 * N - 1, dtype=int)
        x[0] = N
        y = -1 * (scale - np.log2(np.arange(1, 2 * N)) // 1) + scale + 1
        for k in range(1, 2 * N):
            Q = int(np.log2(k) // 1)
            P = int((k) // 2) - 1
            if k > 1:
                if k % 2:
                    x[k - 1] = x[P] + N / (2 ** Q)
                else:
                    x[k - 1] = x[P] - N / (2 ** Q)

        Edges = np.vstack((np.repeat(np.arange(1, N), 2), np.arange(2, 2 * N))).transpose() - 1
        labx = ["".join(k) for k in np.round(HEns, 3).astype(str)]
        ax2 = plt.subplot(G[3:, :])
        for k in range(len(x) - 1):
            ax2.plot(x[Edges[k, :]], y[Edges[k, :]], color="black", zorder=0)
            ax2.annotate(labx[k], (x[k], y[k]), fontsize=8)
        ax2.scatter(
            x,
            y,
            c=HEns,
            zorder=1,
            # Color map and color normalization values
            cmap="spring",
            vmin=colormin,
            vmax=colormax,
        )
        ax2.annotate(labx[-1], (x[-1], y[-1]), fontsize=8)
        ax2.invert_yaxis()
        ax2.set_ylabel("Scale Factor")
        plt.show()

    # return MSx, Sn, CI
    return hen, info


def _hierarchical_decomposition(signal, scale=3):
    N = int(2 ** np.floor(np.log2(len(signal))))
    if N / (2 ** (scale - 1)) < 8:
        raise Exception(
            "Signal length is too short to estimate entropy at the lowest"
            " subtree. Consider reducing the value of scale."
        )

    Q = np.zeros(((2 ** scale) - 1, N))
    Q[0, :] = signal[:N]
    p = 1
    for k in range(scale - 1):
        for n in range(2 ** k):
            Temp = Q[(2 ** k) + n - 1, :]
            # 1. We define an averaging operator Q0. It is the the low frequency component.
            Q[p, : N // 2] = (Temp[::2] + Temp[1::2]) / 2
            # 2. We define a difference frequency component. It is the the high frequency component.
            Q[p + 1, : N // 2] = (Temp[::2] - Temp[1::2]) / 2
            p += 2
    return Q, N
