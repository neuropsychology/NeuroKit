# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def eeg_rereference(eeg, reference="average", robust=False, **kwargs):
    """EEG Rereferencing

    Parameters
    -----------
    eeg : np.ndarray
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    reference : str
        See ``mne.set_eeg_reference()``. Can be a string (e.g., 'average', 'lap' for Laplacian
        "reference-free" transformation, i.e., CSD), or a list (e.g., ['TP9', 'TP10'] for mastoid
        reference).
    robust : bool
        Only applied if reference is 'average'. If True, will substract the median instead of
        the mean.
    **kwargs
        Optional arguments to be passed into ``mne.set_eeg_rereference()``.

    Returns
    -------
    object
        The rereferenced raw mne object.

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
    >>> # Compare the rerefering of an array vs. the MNE object
    >>> data_mne = eeg.copy().set_eeg_reference('average', verbose=False).get_data()
    >>> data_nk = nk.eeg_rereference(eeg.get_data(), 'average')
    >>>
    >>> # Difference between average and LAP
    >>> lap = nk.eeg_rereference(eeg, 'lap')
    >>>
    >>> nk.signal_plot([avg.get_data()[0, 0:1000],
    ...                 lap.get_data()[0, 0:1000]], standardize=True)

    References
    -----------
    - Trujillo, L. T., Stanfield, C. T., & Vela, R. D. (2017). The effect of electroencephalogram (EEG)
    reference choice on information-theoretic measures of the complexity and integration of EEG signals.
    Frontiers in Neuroscience, 11, 425.

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

    eeg = eeg.copy()
    if reference == "average" and robust is True:
        eeg._data = eeg_rereference_array(eeg._data, reference=reference, robust=robust)
        eeg.info["custom_ref_applied"] = True
    elif reference in ["lap", "csd"]:
        try:
            import mne
            if mne.__version__ < '0.20':
                raise ImportError
        except ImportError:
            raise ImportError(
                "NeuroKit error: eeg_rereference(): the 'mne' module (version > 0.20) is required "
                "for this function to run. Please install it first (`pip install mne`).",
            )
        old_verbosity_level = mne.set_log_level(verbose="WARNING", return_old_level=True)
        eeg = mne.preprocessing.compute_current_source_density(eeg)
        mne.set_log_level(old_verbosity_level)
    else:
        eeg = eeg.set_eeg_reference(reference, verbose=False, **kwargs)

    return eeg
