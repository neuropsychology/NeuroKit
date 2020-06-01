import numpy as np

from .signal_zerocrossings import signal_zerocrossings


def signal_decompose(signal):
    """
    >>> import neurokit2 as nk
    >>>
    >>> # Artificial example -----------
    >>> # Create complex signal
    >>> signal = nk.signal_simulate(duration=10, frequency=1, noise=0.01)
    >>> signal += 3 * nk.signal_simulate(duration=10, frequency=3, noise=0.01)
    >>> signal += 3 * np.linspace(0, 2, len(signal))  # Add baseline and trend
    >>> signal += 2 * nk.signal_simulate(duration=10, frequency=0.1, noise=0)
    >>>
    >>> nk.signal_plot(signal)
    >>>
    >>> c = nk.signal_decompose(signal)
    >>>
    >>> # Visualize components and reconstructed signal
    >>> fig = nk.signal_plot(c)
    >>> fig #doctest: +SKIP
    >>> fig2 = nk.signal_plot([signal, np.sum(c, axis=0)])
    >>> fig2 #doctest: +SKIP
    >>>
    >>> # Real example
    >>> ecg = nk.ecg_simulate(duration=60*6, sampling_rate=150)
    >>> signal = nk.ecg_rate(nk.ecg_peaks(ecg, sampling_rate=150), sampling_rate=150)
    >>>
    >>> c = nk.signal_decompose(signal)
    >>>
    >>> # Visualize components and reconstructed signal
    >>> fig = nk.signal_plot(c)
    >>> fig #doctest: +SKIP
    >>>
    >>> fig2 = nk.signal_plot([signal, np.sum(c, axis=0)])
    >>> fig2 #doctest: +SKIP
    """
    components = _signal_decompose_emd(signal, ensemble=False)
    return components


# =============================================================================
# Methods
# =============================================================================
def _signal_decompose_emd(signal, ensemble=False):
    """
    References
    ------------
    - http://perso.ens-lyon.fr/patrick.flandrin/CSDATrendfiltering.pdf
    - https://github.com/laszukdawid/PyEMD
    - https://towardsdatascience.com/decomposing-signal-using-empirical-mode-decomposition-algorithm-explanation-for-dummy-93a93304c541

    >>> # import PyEMD
    >>> # import numpy as np
    >>>
    >>> # signal = np.cos(np.linspace(start=0, stop=10, num=1000))  # Low freq
    >>> # signal += np.cos(np.linspace(start=0, stop=100, num=1000))  # High freq
    >>> # signal += 3  # Add baseline
    >>>
    >>> # emd = PyEMD.EMD()
    >>> # components = emd.emd(signal)
    >>> # imfs, residue = emd.get_imfs_and_residue()
    >>> # nk.signal_plot(imfs)
    >>> # nk.signal_plot([signal, np.sum(imfs, axis=0), residue])
    """
    try:
        import PyEMD
    except ImportError:
        raise ImportError(
            "NeuroKit error: _signal_decompose_emd(): the 'PyEMD' module is required for this function to run. ",
            "Please install it first (`pip install EMD-signal`).",
        )

    if ensemble is False:
        emd = PyEMD.EMD(extrema_detection="parabol")
        imfs = emd.emd(signal)
    else:
        emd = PyEMD.EEMD(extrema_detection="parabol")
        imfs = emd.eemd(signal)

    _, residue = emd.get_imfs_and_residue()
    return imfs


# =============================================================================
# Internals
# =============================================================================


def _signal_decompose_meanfreq(components, sampling_rate=1000):
    duration = components.shape[1] / sampling_rate
    n = len(components)
    freqs = np.zeros(n)

    for i in range(n):
        c = components[i, :] - np.mean(components[i, :])
        freqs[i] = len(signal_zerocrossings(c)) / duration
