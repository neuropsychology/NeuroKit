import numpy as np
import pandas as pd

from .mne_to_df import mne_to_df


def _sanitize_eeg(eeg, sampling_rate=None, time=None):
    """Convert to DataFrame

    Input can be an array (channels, time), or an MNE object.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> # Raw objects
    >>> eeg = nk.mne_data("raw")
    """

    # If array (channels, time), tranpose and convert to DataFrame
    if isinstance(eeg, np.ndarray):
        eeg = pd.DataFrame(eeg.T)
        eeg.columns = [f"EEG_{i}" for i in range(eeg.shape[1])]

    # If dataframe
    if isinstance(eeg, pd.DataFrame):
        return eeg, sampling_rate, time

    # Probably an mne object
    else:
        sampling_rate = eeg.info["sfreq"]
        eeg = mne_to_df(eeg)
        time = eeg["Time"].values
        eeg = eeg.drop(columns=["Time"])

    return eeg, sampling_rate, time
