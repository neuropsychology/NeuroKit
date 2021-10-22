from warnings import warn
import numpy as np
import pandas as pd

from ..stats import standardize
from ..misc import NeuroKitWarning
from ..epochs.eventrelated_utils import _eventrelated_sanitizeinput


def fractal_nld(signal, n_epochs=100, window=None):
    """Fractal dimension of signal epochs via Normalized Length Density (NLD).

    This method was developed for measuring signal complexity on very short epochs durations (i.e., N < 100),
    for when continuous signal FD changes (or 'running' FD) are of interest.

    For methods such as Higuchi's FD, the standard deviation of the window FD increases sharply when the epoch becomes shorter.
    This NLD method results in lower standard deviation especially for shorter epochs,
    though at the expense of lower accuracy in average window FD.

    See Also
    --------
    fractal_higuchi

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    n_epochs : int
        The number of epochs over which the window fractal dimension is computed over.
    window : int, None
        If not None, performs a rolling window standardization, i.e., apply a standardization on a window of the
        specified number of samples that rolls along the main axis of the signal (see ``standardize()`` function).

    Returns
    --------
    fd : DataFrame
        A dataframe containing the fractal dimension across epochs.
    info : dict
        A dictionary containing additional information regarding the parameters used to compute the fractal dimension,
        and the mean and standard deviation of the fractal dimensions.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Get data
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>>
    >>> # Compute FD
    >>> fd, info = nk.fractal_nld(signal, n_epochs=100, window=None)
    >>> fd #doctest: +SKIP

    References
    ----------
    - Kalauzi, A., Bojić, T., & Rakić, L. (2009). Extracting complexity waveforms from one-dimensional signals.
    Nonlinear biomedical physics, 3(1), 1-11.
    - https://github.com/tfburns/MATLAB-functions-for-complexity-measures-of-one-dimensional-signals/blob/master/nld.m
    """

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )
    epochs = np.array_split(signal, n_epochs)

    # Warning
    lengths = list(np.unique([len(i) for i in epochs]))
    if len(lengths) != 1:
        warn(
            f"Computing FD over epochs with unequal lengths {lengths}.",
            category=NeuroKitWarning,
        )

    fd_windows = {}
    for i, epoch in enumerate(epochs):
        fd_windows[i] = {}  # Initialize empty container

        # Add label info
        fd_windows[i]['Length'] = len(epoch)

        # Compute FD
        fd_windows[i]['FD'] = _fractal_nld(epoch, window=window)

    fd = pd.DataFrame.from_dict(fd_windows, orient="index")

    return fd, {'Mean': np.nanmean(fd['FD']), 'SD': np.nanstd(fd['FD']), 'Epochs': n_epochs}


def _fractal_nld(epoch, window=None):

    n = len(epoch)

    # amplitude normalization
    epoch = standardize(epoch, window=window)

    # calculate normalized length density
    nld = np.sum(np.abs(np.diff(epoch))) / n

    # Power model optimal parameters based on analysis of EEG signals (from Kalauzi et al. 2009)
    a = 1.8399
    k = 0.3523
    nld_0 = 0.097178

    # Compute fd
    fd = a * (nld - nld_0) ** k

    return fd
