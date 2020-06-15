# -*- coding: utf-8 -*-

import numpy as np

from ..misc import as_vector
from ..signal import signal_findpeaks


def eog_peaks(eog_cleaned, method="mne"):
    """Locate EOG events (blinks, saccades, eye-movements, ...).

    Prepare a raw EOG signal for eye blinks detection. Only Agarwal & Sivakumar (2019)'s method
    is implemented for now.

    Parameters
    ----------
    eog_cleaned : list or array or Series
        The raw EOG channel.
    method : str
        The peak detection algorithm. Can be one of 'mne' (default) (requires the MNE package
        to be installed).

    Returns
    -------
    array
        Vector containing the samples at which EOG-peaks occur,

    See Also
    --------
    eog_clean

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
    >>> # MNE-method
    >>> mne = nk.eog_peaks(eog_cleaned, method="mne")
    >>> nk.events_plot(mne, eog_cleaned)
    >>>
    >>> # brainstorm method
    >>> brainstorm = nk.eog_peaks(eog_cleaned, method="brainstorm")
    >>> nk.events_plot(brainstorm, eog_cleaned)

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
        peaks = _eog_peaks_mne(eog_cleaned)
    elif method in ["brainstorm"]:
        peaks = _eog_peaks_brainstorm(eog_cleaned)
    else:
        raise ValueError("NeuroKit error: eog_peaks(): 'method' should be " "one of 'mne', 'brainstorm'.")

    return peaks


# =============================================================================
# Methods
# =============================================================================
def _eog_peaks_mne(eog_cleaned):
    """EOG blink detection based on MNE.

    https://github.com/mne-tools/mne-python/blob/master/mne/preprocessing/eog.py

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
        eog_events, _ = mne.preprocessing.peak_finder(eog_cleaned, extrema=1, verbose=False)
    else:
        eog_events, _ = mne.preprocessing.peak_finder(eog_cleaned, extrema=-1, verbose=False)

    return eog_events


def _eog_peaks_brainstorm(eog_cleaned):
    """EOG blink detection implemented in brainstorm.

    https://github.com/mne-tools/mne-python/blob/master/mne/preprocessing/eog.py

    """
    # Find peaks
    peaks = signal_findpeaks(eog_cleaned)["Peaks"]

    # Brainstorm: "An event of interest is detected if the absolute value of the filtered
    # signal value goes over a given number of times the standard deviation. For EOG: 2xStd."
    # -> Remove all peaks that correppond to regions < 2 SD
    mask = eog_cleaned > 2 * np.std(eog_cleaned, ddof=1)
    peaks = peaks[mask[peaks]]

    return peaks
