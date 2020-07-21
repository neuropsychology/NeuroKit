# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal

from ..eeg import eeg_gfp


def microstates_peaks(eeg, gfp=None, sampling_rate=None, distance_between=0.01, **kwargs):
    """Find peaks of stability using the GFP

    Peaks in the global field power (GFP) are often used to find microstates.

    Parameters
    ----------
    eeg : np.ndarray
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    gfp : list
        The Global Field Power (GFP). If None, will be obtained via ``eeg_gfp()``.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    distance_between : float
        The minimum distance (this value is to be multiplied by the sampling rate) between peaks.
        The default is 0.01, which corresponds to 10 ms (as suggested in the Microstate EEGlab
        toolbox).
    **kwargs
        Additional arguments to be passed to ``eeg_gfp()``.

    Returns
    -------
    peaks : array
        The index of the sample where GFP peaks occur.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> eeg = nk.mne_data("filt-0-40_raw")
    >>>
    >>> gfp = nk.eeg_gfp(eeg)
    >>> peaks1 = nk.microstates_peaks(eeg, distance_between=0.01)
    >>> peaks2 = nk.microstates_peaks(eeg, distance_between=0.05)
    >>> peaks3 = nk.microstates_peaks(eeg, distance_between=0.10)
    >>> nk.events_plot([peaks1[peaks1 < 500],
    ...                 peaks2[peaks2 < 500],
    ...                 peaks3[peaks3 < 500]], gfp[0:500]) #doctest: +ELLIPSIS
    <Figure ...>

    See Also
    --------
    eeg_gfp

    """
    if isinstance(eeg, (pd.DataFrame, np.ndarray)) is False:
        sampling_rate = eeg.info["sfreq"]
        eeg = eeg.get_data()

    if sampling_rate is None:
        raise ValueError("NeuroKit error: microstates_peaks(): The sampling_rate is requested ",
                         "for this function to run. Please provide it as an argument.")

    # If we want ALL the indices
    if gfp is False:
        return np.arange(len(eeg))

    # If we don't want to rely on peaks but take uniformly spaced samples (used in microstates_clustering)
    if isinstance(gfp, (int, float, str)):
        if isinstance(gfp, str):  # If gfp = 'all'
            gfp = len(eeg[0, :])
        if gfp <= 1:  # If fraction
            gfp = np.int(gfp * len(eeg[0, :]))
        return np.linspace(0, len(eeg[0, :]), gfp, endpoint=False, dtype=np.int)

    # If GFP peaks
    if gfp is None:
        gfp = eeg_gfp(eeg, **kwargs)

    peaks = _microstates_peaks_gfp(gfp=gfp, sampling_rate=sampling_rate, distance_between=distance_between)

    return peaks


# =============================================================================
# Methods
# =============================================================================
def _microstates_peaks_gfp(gfp=None, sampling_rate=None, distance_between=0.01):

    minimum_separation = int(distance_between * sampling_rate)  # 10 ms (Microstate EEGlab toolbox)
    if minimum_separation == 0:
        minimum_separation = 1

    peaks_gfp, _ = scipy.signal.find_peaks(gfp, distance=minimum_separation)

    # Alternative methods: (doesn't work best IMO)
#    peaks_gfp = scipy.signal.find_peaks_cwt(gfp, np.arange(minimum_separation, int(0.2 * sampling_rate)))
#    peaks_gfp = scipy.signal.argrelmax(gfp)[0]

    # Use DISS
#    diss = nk.eeg_diss(eeg, gfp)
#    peaks_diss, _ = scipy.signal.find_peaks(diss, distance=minimum_separation)

    return peaks_gfp
