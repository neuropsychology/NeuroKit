# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..stats import mad
from ..signal import signal_filter


def eeg_rereference(eeg, reference="average", robust=False, **kwargs):
    """EEG Rereferencing

    Parameters
    -----------
    eeg : np.array
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    reference : str
        See ``mne.set_eeg_reference()``. Most common references include 'average'.
    robust : bool
        If True and reference is 'average', will substract the median instead of the mean.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> eeg = nk.mne_data("filt-0-40_raw")
    >>>
    >>> # Difference between robust average
    >>> avg = nk.eeg_rereference(eeg, 'average', robust=False)
    >>> avg_r = nk.eeg_rereference(eeg, 'average', robust=True)
    >>>
    >>> nk.signal_plot([avg.get_data()[0, 0:1000],
    ...                 avg_r.get_data()[0, 0:1000]])
    >>>
    >>> # Compare the rerefering of an array to MNE
    >>> data_mne = eeg.copy().set_eeg_reference('average', verbose=False).get_data()
    >>> data_nk = nk.eeg_rereference(eeg.get_data(), 'average')
    >>> np.all(data_mne == data_nk)

    """
    # If MNE object
    if isinstance(eeg, (pd.DataFrame, np.ndarray)):
        eeg = eeg_rereference_array(eeg, reference=reference, robust=robust)
    else:
        eeg = eeg_rereference_mne(eeg, reference=reference, robust=robust, **kwargs)
    return eeg



# =============================================================================
# Methods
# =============================================================================
def eeg_rereference_array(eeg, reference="average", robust=False):
    """
    """
    # Average reference
    if reference == "average":
        if robust is False:
            eeg = eeg - np.mean(eeg, axis=0, keepdims=True)
        else:
            eeg = eeg - np.median(eeg, axis=0, keepdims=True)
    else:
        raise ValueError("NeuroKit error: eeg_rereference(): Only 'average' rereferencing",
                         " is supported for data arrays for now.")

    return eeg

def eeg_rereference_mne(eeg, reference="average", robust=False, **kwargs):
    """
    """
    eeg = eeg.copy()
    if reference == "average" and robust is True:
        eeg._data = eeg_rereference_array(eeg._data, reference=reference, robust=robust)
        eeg.info["custom_ref_applied"] = True
    else:
        eeg = eeg.set_eeg_reference(reference, verbose=False, **kwargs)

    return eeg