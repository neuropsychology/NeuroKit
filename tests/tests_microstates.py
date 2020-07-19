# -*- coding: utf-8 -*-
import mne
import numpy as np

import neurokit2 as nk

# =============================================================================
# Peaks
# =============================================================================


def test_microstates_peaks():

    # Load eeg data and calculate gfp
    eeg = nk.mne_data("filt-0-40_raw")
    gfp = nk.eeg_gfp(eeg)

    # Find peaks
    peaks_nk = nk.microstates_peaks(eeg, distance_between=0.01)

    # Test with alternative method taken from Frederic
    # https://github.com/Frederic-vW/eeg_microstates/blob/master/eeg_microstates.py
    def locmax(x):
        dx = np.diff(x)  # discrete 1st derivative
        zc = np.diff(np.sign(dx))  # zero-crossings of dx
        m = 1 + np.where(zc == -2)[0]  # indices of local max.
        return m

    peaks_frederic = locmax(gfp)

    assert all(elem in peaks_frederic for elem in peaks_nk)  # only works when distance_between = 0.01
