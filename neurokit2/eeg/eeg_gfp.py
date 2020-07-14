# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..stats import mad


def eeg_gfp(eeg, normalize=False, robust=False):
    """Global Field Power (GFP)

    Global Field Power (GFP) constitutes a reference-independent measure of response strength.
    GFP was first introduced by Lehmann and Skrandies (1980) and has since become a commonplace
    measure among M/EEG users. Mathematically, GFP is the standard deviation of all electrodes
    at a given time

    Parameters
    ----------
    eeg : np.array
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    normalize : bool
        Should the data by standardized (z-score) the data across time prior to GFP extraction.
    robust : bool
        If True, the normalization and the GFP extraction will be done using the median/MAD instead
        of the mean/SD.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> eeg = nk.mne_data("filt-0-40_raw")
    >>> eeg = eeg.set_eeg_reference('average')
    >>>
    >>> gfp = nk.eeg_gfp(eeg)
    >>> gfp_z = nk.eeg_gfp(eeg, normalize=True)
    >>> gfp_zr = nk.eeg_gfp(eeg, normalize=True, robust=True)
    >>> nk.signal_plot([gfp[0:500], gfp_z[0:500], gfp_zr[0:500]], standardize=True)
    >>>

    References
    ----------
    - Lehmann, D., & Skrandies, W. (1980). Reference-free identification of components of
    checkerboard-evoked multichannel potential fields. Electroencephalography and clinical
    neurophysiology, 48(6), 609-621.

    """
    if isinstance(eeg, (pd.DataFrame, np.ndarray)) is False:
        eeg = eeg.get_data()

    # Average reference
    if robust is True:
        eeg = eeg - np.mean(eeg, axis=0, keepdims=True)
    else:
        eeg = eeg - np.median(eeg, axis=0, keepdims=True)

    # Normalization
    if normalize is True:
        if robust is True:
            eeg = eeg / np.std(eeg, axis=0, ddof=0)
        else:
            eeg = eeg / mad(eeg, axis=0)

    # Compute GFP
    if robust is True:
        gfp = mad(eeg, axis=0)
    else:
        gfp = np.std(eeg, axis=0, ddof=0)

    return gfp

