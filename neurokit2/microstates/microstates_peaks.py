# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal

def microstates_peaks(eeg, sampling_rate=1000, normalize=True, robust=False):
    """Find peaks of stability using the GFP

    Microstate boarders were determined by relative minima of GFP together with relative maxima in Diss.

    https://www.researchgate.net/publication/7432398_Response_inhibition_deficits_in_externalizing_child_psychiatric_disorders_An_ERP-study_with_the_Stop-task

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> eeg = nk.mne_data("filt-0-40_raw")
    >>>

    """
    gfp = nk.eeg_gfp(eeg, normalize=normalize, robust=robust)
    diss = nk.eeg_diss(eeg, gfp)

    minimum_separation = int(0.005 * sampling_rate)
    if minimum_separation == 0:
        minimum_separation = 1
    peaks_gfp, _ = scipy.signal.find_peaks(-gfp, distance=minimum_separation)
    peaks_diss, _ = scipy.signal.find_peaks(diss, distance=minimum_separation)

