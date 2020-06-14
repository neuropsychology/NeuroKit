# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ..misc import as_vector
from ..signal import signal_filter


def eog_peaks(eog_cleaned, method="mne"):
    """Locate EOG events (blinks, saccades, eye-movements, ...).

    Prepare a raw EOG signal for eye blinks detection. Only Agarwal & Sivakumar (2019)'s method
    is implemented for now.

    Parameters
    ----------
    eog_signal : list or array or Series
        The raw EOG channel.
    method : str
        The peak detection algorithm. Can be one of 'mne' (default) (requires the MNE package
        to be installed).

    See Also
    --------
    eog_clean

    Examples
    --------
    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Get data
    >>> eog_signal = nk.data('eog_100hz')["vEOG"]
    >>>
    >>> # Clean
    >>> eog_cleaned = nk.eog_clean(eog_signal, sampling_rate=100, method='mne')
    >>>
    >>> # Find peaks
    >>> peaks = nk.eog_peaks(eog_cleaned)
    >>>
    >>> # Visualize
    >>> nk.events_plot(peaks, eoc_cleaned)

    References
    ----------
    - Agarwal, M., & Sivakumar, R. (2019). Blink: A Fully Automated Unsupervised Algorithm for
    Eye-Blink Detection in EEG Signals. In 2019 57th Annual Allerton Conference on Communication,
    Control, and Computing (Allerton) (pp. 1113-1121). IEEE.

    """
    # Sanitize input
    eog_cleaned = as_vector(eog_cleaned)

    # Apply method
    method = method.lower()
    if method in ["mne"]:
        peaks = _eog_peaks_mne(eog_cleaned, sampling_rate=sampling_rate)
    else:
        raise ValueError(
            "NeuroKit error: eog_peaks(): 'method' should be "
            "one of 'mne'."
        )


    return peaks


# =============================================================================
# Methods
# =============================================================================
def _eog_peaks_mne(eog_cleaned, sampling_rate=1000):
    """https://github.com/mne-tools/mne-python/blob/master/mne/preprocessing/eog.py
    """
    # Make sure MNE is installed
    try:
        import mne
    except ImportError:
        raise ImportError(
            "NeuroKit error: signal_filter(): the 'mne' module is required for this method to run. ",
            "Please install it first (`pip install mne`).",
        )

    # Find peaks
    temp = eog_cleaned - np.mean(eog_cleaned)

    if np.abs(np.max(temp)) > np.abs(np.min(temp)):
        eog_events, _ = mne.preprocessing.peak_finder(eog_cleaned, extrema=1)
    else:
        eog_events, _ = mne.preprocessing.peak_finder(eog_cleaned, extrema=-1)

    return eog_events
