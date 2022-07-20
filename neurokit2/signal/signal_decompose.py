import numpy as np

from ..misc import as_vector


def signal_decompose(signal, method="emd", n_components=None, **kwargs):
    """**Decompose a signal**

    Signal decomposition into different sources using different methods, such as Empirical Mode
    Decomposition (EMD) or Singular spectrum analysis (SSA)-based signal separation method.

    The extracted components can then be recombined into meaningful sources using
    :func:`.signal_recompose`.

    Parameters
    -----------
    signal : Union[list, np.array, pd.Series]
        Vector of values.
    method : str
        The decomposition method. Can be one of ``"emd"`` or ``"ssa"``.
    n_components : int
        Number of components to extract. Only used for ``"ssa"`` method. If ``None``, will default
        to 50.
    **kwargs
        Other arguments passed to other functions.

    Returns
    -------
    Array
        Components of the decomposed signal.

    See Also
    --------
    signal_recompose

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Create complex signal
      signal = nk.signal_simulate(duration=10, frequency=1, noise=0.01)  # High freq
      signal += 3 * nk.signal_simulate(duration=10, frequency=3, noise=0.01)  # Higher freq
      signal += 3 * np.linspace(0, 2, len(signal))  # Add baseline and trend
      signal += 2 * nk.signal_simulate(duration=10, frequency=0.1, noise=0)

      @savefig p_signal_decompose1.png scale=100%
      nk.signal_plot(signal)
      @suppress
      plt.close()

    .. ipython:: python
      :okexcept:

      # Example 1: Using the EMD method
      components = nk.signal_decompose(signal, method="emd")

      # Visualize Decomposed Signal Components
      @savefig p_signal_decompose2.png scale=100%
      nk.signal_plot(components)
      @suppress
      plt.close()

    .. ipython:: python

      # Example 2: USing the SSA method
      components = nk.signal_decompose(signal, method="ssa", n_components=5)

      # Visualize Decomposed Signal Components
      @savefig p_signal_decompose3.png scale=100%
      nk.signal_plot(components)  # Visualize components
      @suppress
      plt.close()

    """
    # Apply method
    method = method.lower()
    if method in ["emd"]:
        components = _signal_decompose_emd(signal, **kwargs)
    elif method in ["ssa"]:
        components = _signal_decompose_ssa(signal, n_components=n_components)
    else:
        raise ValueError(
            "NeuroKit error: signal_decompose(): 'method' should be one of 'emd' or 'ssa'."
        )
    return components


# =============================================================================
# Singular spectrum analysis (SSA)
# =============================================================================
def _signal_decompose_ssa(signal, n_components=None):
    """Singular spectrum analysis (SSA)-based signal separation method.

    SSA decomposes a time series into a set of summable components that are grouped together and
    interpreted as trend, periodicity and noise.

    References
    ----------
    - https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition

    """
    # sanitize input
    signal = as_vector(signal)

    # Parameters
    # The window length.
    if n_components is None:
        L = 50 if len(signal) >= 100 else int(len(signal) / 2)
    else:
        L = n_components

    # Length.
    N = len(signal)
    if not 2 <= L <= N / 2:
        raise ValueError("`n_components` must be in the interval [2, len(signal)/2].")

    # The number of columns in the trajectory matrix.
    K = N - L + 1

    # Embed the time series in a trajectory matrix by pulling the relevant subseries of F,
    # and stacking them as columns.
    X = np.array([signal[i : L + i] for i in range(0, K)]).T

    # Get n components
    d = np.linalg.matrix_rank(X)

    # Decompose the trajectory matrix
    u, sigma, vt = np.linalg.svd(X, full_matrices=False)

    # Initialize components matrix
    components = np.zeros((N, d))
    # Reconstruct the elementary matrices without storing them
    for i in range(d):
        X_elem = sigma[i] * np.outer(u[:, i], vt[i, :])
        X_rev = X_elem[::-1]
        components[:, i] = [
            X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])
        ]

    # Return the components
    return components.T


# =============================================================================
# ICA
# =============================================================================
# import sklearn.decomposition
# def _signal_decompose_scica(signal, n_components=3, **kwargs):
#    # sanitize input
#    signal = as_vector(signal)
#
#    # Single-channel ICA (SCICA)
#    if len(signal.shape) == 1:
#        signal = signal.reshape(-1, 1)
#
#    c = sklearn.decomposition.FastICA(n_components=n_components, **kwargs).fit_transform(signal)


# =============================================================================
# Empirical Mode Decomposition (EMD)
# =============================================================================
def _signal_decompose_emd(signal, ensemble=False, **kwargs):
    """References
    ------------
    - http://perso.ens-lyon.fr/patrick.flandrin/CSDATrendfiltering.pdf
    - https://github.com/laszukdawid/PyEMD
    - https://towardsdatascience.com/decomposing-signal-using-empirical-mode-decomposition-algorithm-explanation-for-dummy-93a93304c541 # noqa: E501

    """
    try:
        import PyEMD
    except ImportError as e:
        raise ImportError(
            "NeuroKit error: _signal_decompose_emd(): the 'PyEMD' module is required for this"
            " function to run. Please install it first (`pip install EMD-signal`).",
        ) from e

    if ensemble is False:
        emd = PyEMD.EMD(extrema_detection="parabol", **kwargs)
        imfs = emd.emd(signal, **kwargs)
    else:
        emd = PyEMD.EEMD(extrema_detection="parabol", **kwargs)
        imfs = emd.eemd(signal, **kwargs)

    #    _, residue = emd.get_imfs_and_residue()
    return imfs
