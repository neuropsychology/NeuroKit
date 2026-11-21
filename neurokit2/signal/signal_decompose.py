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
        Only used for the ``"ssa"`` method, where it is the window length, i.e., the maximum number
        of components (fewer are returned if the signal has a lower rank). If ``None``, defaults to
        50, or to half the signal length for signals shorter than 100 samples.
    **kwargs
        Other arguments. For the ``"emd"`` method, ``ensemble=True`` runs Ensemble EMD, and the
        others are passed to PyEMD (e.g., ``max_imf``, ``std_thr``, or ``extrema_detection``, which
        defaults to ``"parabol"``).

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
      :okwarning:

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
    signal = as_vector(signal)
    if method in ["emd"]:
        components = _signal_decompose_emd(signal, **kwargs)
    elif method in ["ssa"]:
        components = _signal_decompose_ssa(signal, n_components=n_components)
    else:
        raise ValueError("NeuroKit error: signal_decompose(): 'method' should be one of 'emd' or 'ssa'.")
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
    X = np.lib.stride_tricks.sliding_window_view(signal, L).T

    # Get n components
    d = np.linalg.matrix_rank(X)

    # Decompose the trajectory matrix
    u, sigma, vt = np.linalg.svd(X, full_matrices=False)

    # Initialize components matrix
    components = np.zeros((d, N))
    counts = np.convolve(np.ones(L), np.ones(K))
    for i in range(d):
        components[i] = sigma[i] * np.convolve(u[:, i], vt[i, :]) / counts

    # Return the components
    return components


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
def _signal_decompose_emd(signal, ensemble=False, T=None, max_imf=-1, progress=False, extrema_detection="parabol", **kwargs):
    """References
    ------------
    - http://perso.ens-lyon.fr/patrick.flandrin/CSDATrendfiltering.pdf
    - https://github.com/laszukdawid/PyEMD
    - https://towardsdatascience.com/decomposing-signal-using-empirical-mode-decomposition-algorithm-explanation-for-dummy-93a93304c541

    """  # noqa: E501
    try:
        import PyEMD
    except ImportError as e:
        raise ImportError(
            "NeuroKit error: _signal_decompose_emd(): the 'PyEMD' module is required for this"
            " function to run. Please install it first (`pip install EMD-signal`).",
        ) from e

    if ensemble is False:
        emd = PyEMD.EMD(extrema_detection=extrema_detection, **kwargs)
        imfs = emd.emd(signal, T=T, max_imf=max_imf)
    else:
        emd = PyEMD.EEMD(extrema_detection=extrema_detection, **kwargs)
        imfs = emd.eemd(signal, T=T, max_imf=max_imf, progress=progress)

    #    _, residue = emd.get_imfs_and_residue()
    return imfs
