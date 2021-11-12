from warnings import warn

import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from ..stats import standardize


def fractal_nld(signal, window=30):
    """Fractal dimension of signal epochs via Normalized Length Density (NLD).

    This method was developed for measuring signal complexity on very short epochs durations (< 30 samples),
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
    window : int
        The duration of the epochs (in samples) by which to cut the signal. Default to 30 samples.

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
    >>> fd, info = nk.fractal_nld(signal)
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

    # Split signal into epochs
    n_epochs = len(signal) // window
    epochs = np.array_split(signal, n_epochs)

    fds = [_fractal_nld(i) for i in epochs]

    return np.nanmean(fds), {"SD": np.nanstd(fds), "Values": fds}


# =============================================================================
# Utilities
# =============================================================================
def _fractal_nld(epoch):

    n = len(epoch)

    # amplitude normalization
    epoch = standardize(epoch)

    # calculate normalized length density
    nld = np.sum(np.abs(np.diff(epoch))) / n

    # Power model optimal parameters based on analysis of EEG signals (from Kalauzi et al. 2009)
    a = 1.8399
    k = 0.3523
    nld_0 = 0.097178

    if (nld - nld_0) < 0:
        warn(
            "Normalized Length Density of some epochs (reflected as `np.nan`in `Values`) may be too short.",
            category=NeuroKitWarning,
        )

    # Compute fd
    return a * (nld - nld_0) ** k
