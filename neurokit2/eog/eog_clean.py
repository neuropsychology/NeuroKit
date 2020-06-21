# -*- coding: utf-8 -*-
import numpy as np
import scipy.ndimage

from ..misc import as_vector
from ..signal import signal_filter


def eog_clean(eog_signal, sampling_rate=1000, method="neurokit"):
    """Clean an EOG signal.

    Prepare a raw EOG signal for eye blinks detection.

    Parameters
    ----------
    eog_signal : Union[list, np.array, pd.Series]
        The raw EOG channel (either vertical or horizontal).
    sampling_rate : int
        The sampling frequency of `eog_signal` (in Hz, i.e., samples/second).
        Defaults to 1000.
    method : str
        The processing pipeline to apply. Can be one of 'neurokit' (default), 'agarwal2019',
        'mne' (requires the MNE package to be installed), 'brainstorm', 'kong1998'.

    Returns
    -------
    array
        Vector containing the cleaned EOG signal.

    See Also
    --------
    signal_filter, eog_peaks

    Examples
    --------
    Examples
    --------
    >>> import neurokit2 as nk
    >>> import pandas as pd
    >>>
    >>> # Get data
    >>> eog_signal = nk.data('eog_100hz')
    >>>
    >>> # Clean
    >>> neurokit = nk.eog_clean(eog_signal, sampling_rate=100, method='neurokit')
    >>> kong1998 = nk.eog_clean(eog_signal, sampling_rate=100, method='kong1998')
    >>> agarwal2019 = nk.eog_clean(eog_signal, sampling_rate=100, method='agarwal2019')
    >>> mne = nk.eog_clean(eog_signal, sampling_rate=100, method='mne')
    >>> brainstorm = nk.eog_clean(eog_signal, sampling_rate=100, method='brainstorm')
    >>> blinker = nk.eog_clean(eog_signal, sampling_rate=100, method='blinker')
    >>>
    >>> # Visualize
    >>> fig = pd.DataFrame({"Raw": eog_signal["vEOG"],
    ...                     "neurokit": neurokit,
    ...                     "kong1998": kong1998,
    ...                     "agarwal2019": agarwal2019,
    ...                     "mne": mne,
    ...                     "brainstorm": brainstorm,
    ...                     "blinker": blinker}).plot(subplots=True)  #doctest: +ELLIPSIS
    <matplotlib.axes._subplots.AxesSubplot at ...>


    References
    ----------
    - Agarwal, M., & Sivakumar, R. (2019). Blink: A Fully Automated Unsupervised Algorithm for
    Eye-Blink Detection in EEG Signals. In 2019 57th Annual Allerton Conference on Communication,
    Control, and Computing (Allerton) (pp. 1113-1121). IEEE.
    - Kleifges, K., Bigdely-Shamlo, N., Kerick, S. E., & Robbins, K. A. (2017). BLINKER: automated
    extraction of ocular indices from EEG enabling large-scale analysis. Frontiers in neuroscience,
    11, 12.
    - Kong, X., & Wilson, G. F. (1998). A new EOG-based eyeblink detection algorithm.
    Behavior Research Methods, Instruments, & Computers, 30(4), 713-719.

    """
    # Sanitize input
    eog_signal = as_vector(eog_signal)

    # Apply method
    method = method.lower()
    if method in ["neurokit", "nk"]:
        clean = _eog_clean_neurokit(eog_signal, sampling_rate=sampling_rate)
    elif method in ["agarwal", "agarwal2019"]:
        clean = _eog_clean_agarwal2019(eog_signal, sampling_rate=sampling_rate)
    elif method in ["brainstorm"]:
        clean = _eog_clean_brainstorm(eog_signal, sampling_rate=sampling_rate)
    elif method in ["mne"]:
        clean = _eog_clean_mne(eog_signal, sampling_rate=sampling_rate)
    elif method in ["blinker", "kleifges2017", "kleifges"]:
        clean = _eog_clean_blinker(eog_signal, sampling_rate=sampling_rate)
    elif method in ["kong1998", "kong"]:
        clean = _eog_clean_kong1998(eog_signal, sampling_rate=sampling_rate)
    else:
        raise ValueError(
            "NeuroKit error: eog_clean(): 'method' should be one of 'agarwal2019', 'brainstorm',",
            "'mne', 'kong1998', 'blinker'.",
        )

    return clean


# =============================================================================
# Methods
# =============================================================================
def _eog_clean_neurokit(eog_signal, sampling_rate=1000):
    """NeuroKit method."""
    return signal_filter(
        eog_signal, sampling_rate=sampling_rate, method="butterworth", order=6, lowcut=0.25, highcut=7.5
    )


def _eog_clean_agarwal2019(eog_signal, sampling_rate=1000):
    """Agarwal, M., & Sivakumar, R.

    (2019). Blink: A Fully Automated Unsupervised Algorithm for Eye-Blink Detection in EEG Signals. In 2019 57th
    Annual Allerton Conference on Communication, Control, and Computing (Allerton) (pp. 1113-1121). IEEE.

    """
    return signal_filter(
        eog_signal, sampling_rate=sampling_rate, method="butterworth", order=4, lowcut=None, highcut=10
    )


def _eog_clean_brainstorm(eog_signal, sampling_rate=1000):
    """EOG cleaning implemented by default in Brainstorm.

    https://neuroimage.usc.edu/brainstorm/Tutorials/TutRawSsp

    """
    return signal_filter(eog_signal, sampling_rate=sampling_rate, method="butterworth", order=4, lowcut=1.5, highcut=15)


def _eog_clean_blinker(eog_signal, sampling_rate=1000):
    """Kleifges, K., Bigdely-Shamlo, N., Kerick, S.

    E., & Robbins, K. A. (2017). BLINKER: automated extraction of ocular indices from EEG enabling large-scale
    analysis. Frontiers in neuroscience, 11, 12.

    """
    # "Each candidate signal is band-passed filtered in the interval [1, 20] Hz prior
    # to blink detection."
    return signal_filter(eog_signal, sampling_rate=sampling_rate, method="butterworth", order=4, lowcut=1, highcut=20)


def _eog_clean_mne(eog_signal, sampling_rate=1000):
    """EOG cleaning implemented by default in MNE.

    https://github.com/mne-tools/mne-python/blob/master/mne/preprocessing/eog.py

    """
    # Make sure MNE is installed
    try:
        import mne
    except ImportError:
        raise ImportError(
            "NeuroKit error: signal_filter(): the 'mne' module is required for this method to run.",
            " Please install it first (`pip install mne`).",
        )

    # Filter
    clean = mne.filter.filter_data(
        eog_signal,
        sampling_rate,
        l_freq=1,
        h_freq=10,
        filter_length="10s",
        l_trans_bandwidth=0.5,
        h_trans_bandwidth=0.5,
        phase="zero-double",
        fir_window="hann",
        fir_design="firwin2",
        verbose=False,
    )

    return clean


def _eog_clean_kong1998(eog_signal, sampling_rate=1000):
    """Kong, X., & Wilson, G.

    F. (1998). A new EOG-based eyeblink detection algorithm. Behavior Research Methods, Instruments, & Computers,
    30(4), 713-719.

    """
    #  The order E should be less than half of the expected eyeblink duration. For example, if
    # the expected blink duration is 200 msec (10 samples with a sampling rate of 50 Hz), the
    # order E should be less than five samples.
    eroded = scipy.ndimage.grey_erosion(eog_signal, size=int((0.2 / 2) * sampling_rate))

    # a "low-noise" Lanczos differentiation filter introduced in Hamming (1989) is employed.
    # Frequently, a first order differentiation filter is sufficient and has the familiar
    # form of symmetric difference:
    # w[k] = 0.5 * (y[k + 1] - y[k - 1])
    diff = eroded - np.concatenate([[0], 0.5 * np.diff(eroded)])

    # To reduce the effects of noise, characterized by small fluctuations around zero, a
    # median filter is also used with the order of the median filter denoted as M.
    # The median filter acts like a mean filter except that it preserves the sharp edges ofthe
    # input. The order M should be less than a quarter ofthe expected eyeblink duration.
    clean = scipy.ndimage.median_filter(diff, size=int((0.2 / 4) * sampling_rate))

    return clean
