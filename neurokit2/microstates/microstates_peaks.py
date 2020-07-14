# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal

from ..eeg import eeg_gfp


def microstates_peaks(eeg, gfp=None, sampling_rate=None, distance_between=0.01, **kwargs):
    """Find peaks of stability using the GFP

    Microstate boarders were determined by relative minima of GFP together with relative maxima in Diss.

    https://www.researchgate.net/publication/7432398_Response_inhibition_deficits_in_externalizing_child_psychiatric_disorders_An_ERP-study_with_the_Stop-task

    Parameters
    ----------
    distance_between : float
        The minimum distance (this value is to be multiplied by the sampling rate) between peaks.
        The default is 0.01, which corresponds to 10 ms (as suggested in the Microstate EEGlab
        toolbox).
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
    ...                 peaks3[peaks3 < 500]], gfp[0:500])

    """
    if isinstance(eeg, (pd.DataFrame, np.ndarray)) is False:
        sampling_rate = eeg.info["sfreq"]
        eeg = eeg.get_data()

    if sampling_rate is None:
        raise ValueError("NeuroKit error: microstates_peaks(): The sampling_rate is requested ",
                         "for this function to run. Please provide it as an argument.")

    if gfp is False:
        return np.arange(len(eeg))

    if gfp is None:
        gfp = eeg_gfp(eeg, **kwargs)

    peaks = _microstates_peaks_gfp(eeg, gfp=None, sampling_rate=None, distance_between=0.01)
    return peaks





def _microstates_peaks_gfp(eeg, gfp=None, sampling_rate=None, distance_between=0.01):

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
