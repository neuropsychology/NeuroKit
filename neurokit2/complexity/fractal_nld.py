import numpy as np

from ..stats import standardize
from ..misc import NeuroKitWarning
from ..epochs.eventrelated_utils import _eventrelated_sanitizeinput


def fractal_nld(epochs, what='ECG', window=None):
    """Fractal dimension of a signal epoch via Normalized Length Density (NLD).

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
    epochs : Union[dict, pd.DataFrame]
        A dict containing one DataFrame per event/trial,
        usually obtained via `epochs_create()`, or a DataFrame
        containing all epochs, usually obtained via `epochs_to_df()`.
    what : str
        The signal to extract from the epochs.
    window : int, None
        If not None, performs a rolling window standardization, i.e., apply a standardization on a window of the
        specified number of samples that rolls along the main axis of the signal.

    Returns
    --------
    fd : DataFrame
        A dataframe containing the fractal dimension across epochs.
    info : dict
        A dictionary containing additional information regarding the parameters used to compute the fractal dimension.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Get data
    >>> data = nk.data("bio_eventrelated_100hz")
    >>> # Find events
    >>> events = nk.events_find(data["Photosensor"],
    ...                         threshold_keep='below',
    ...                         event_conditions=["Negative", "Neutral", "Neutral", "Negative"])
    >>> # Create epochs
    >>> epochs = nk.epochs_create(data, events, sampling_rate=100, epochs_start=-0.5, epochs_end=0.2)
    >>>
    >>> # Compute FD
    >>> nld, _ = nk.fractal_nld(epochs, what='ECG', window=None)

    References
    ----------
    - Kalauzi, A., Bojić, T., & Rakić, L. (2009). Extracting complexity waveforms from one-dimensional signals.
    Nonlinear biomedical physics, 3(1), 1-11.
    - https://github.com/tfburns/MATLAB-functions-for-complexity-measures-of-one-dimensional-signals/blob/master/nld.m
    """

    # Sanity checks
    epochs = _eventrelated_sanitizeinput(epochs, what=what, silent=False)

    # Sanitize input
    if what is None:
        raise ValueError(
            "Please specify the signal (column name, e.g., 'ECG') that you want to compute FD over.",
            category=NeuroKitWarning
        )
        return output

    fd_windows = {}
    for index in epochs:
        fd_windows[index] = {}  # Initialize empty container

        # Add label info
        fd_windows[index]['Epoch'] = epochs[index]['Label'].iloc[0]

        # Add label info
        fd_windows[index]['Sample_Start'] = epochs[index]['Index'].iloc[0]

        # Compute FD
        fd_windows[index]['FD'] = _fractal_nld(epochs[index][what], window=window)

    fd = pd.DataFrame.from_dict(fd_windows, orient="index")

    return fd, {'Window': window}


def _fractal_nld(epoch, window=None):

    n = len(epoch)

    # amplitude normalization
    epoch = standardize(np.array(epoch), window=window)

    # calculate normalized length density
    nld = np.sum(np.abs(np.diff(epoch))) / n

    # Power model optimal parameters based on analysis of EEG signals (from Kalauzi et al. 2009)
    a = 1.8399
    k = 0.3523
    nld_0 = 0.097178

    # Compute fd
    fd = a * (nld - nld_0) ** k

    return fd
