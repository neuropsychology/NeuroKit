# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..stats import mad
from ..signal import signal_filter


def eeg_gfp(eeg, sampling_rate=None, normalize=False, robust=False, method="l1", smooth=0):
    """Global Field Power (GFP)

    Global Field Power (GFP) constitutes a reference-independent measure of response strength.
    GFP was first introduced by Lehmann and Skrandies (1980) and has since become a commonplace
    measure among M/EEG users. Mathematically, GFP is the standard deviation of all electrodes
    at a given time

    Parameters
    ----------
    eeg : np.array
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second). Only necessary if
        smoothing is requested.
    normalize : bool
        Should the data by standardized (z-score) the data across time prior to GFP extraction.
    robust : bool
        If True, the normalization and the GFP extraction will be done using the median/MAD instead
        of the mean/SD.
    method : str
        Can be either 'l1' or 'l2' to use the L1 or L2 norm.
    smooth : float
        Can be either None or a float. If a float, will use this value, multiplied by the
        sampling rate

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
    >>> gfp_s = nk.eeg_gfp(eeg, smooth=0.05)
    >>> nk.signal_plot([gfp[0:500], gfp_z[0:500], gfp_zr[0:500], gfp_s[0:500]], standardize=True)

    References
    ----------
    - Lehmann, D., & Skrandies, W. (1980). Reference-free identification of components of
    checkerboard-evoked multichannel potential fields. Electroencephalography and clinical
    neurophysiology, 48(6), 609-621.

    """
    # If MNE object
    if isinstance(eeg, (pd.DataFrame, np.ndarray)) is False:
        sampling_rate = eeg.info["sfreq"]
        eeg = eeg.get_data()

    # Average reference
    if robust is False:
        eeg = eeg - np.mean(eeg, axis=0, keepdims=True)
    else:
        eeg = eeg - np.median(eeg, axis=0, keepdims=True)

    # Normalization
    if normalize is True:
        if robust is False:
            eeg = eeg / np.std(eeg, axis=0, ddof=0)
        else:
            eeg = eeg / mad(eeg, axis=0)

    # Compute GFP
    if method.lower() == "l1":
        gfp = _eeg_gfp_L1(eeg, robust=robust)
    else:
        gfp = _eeg_gfp_L2(eeg, robust=robust)

    # Smooth
    if smooth is not None and smooth != 0:
        gfp = _eeg_gfp_smoothing(gfp, sampling_rate=sampling_rate, window_size=smooth)

    return gfp


# =============================================================================
# Utilities
# =============================================================================
def _eeg_gfp_smoothing(gfp, sampling_rate=None, window_size=0.02):
    """
    Smooth the Global Field Power Curve
    """
    if sampling_rate is None:
        raise ValueError("NeuroKit error: eeg_gfp(): You requested to smooth the GFP, for which ",
                         "we need to know the sampling_rate. Please provide it as an argument.")
    window = int(window_size * sampling_rate)
    if window > 2:
        gfp = signal_filter(gfp, method="savgol", order=2, window_size=window)

    return gfp


# =============================================================================
# Methods
# =============================================================================

def _eeg_gfp_L1(eeg, robust=False):
    if robust is False:
        gfp = np.sum(np.abs(eeg - np.mean(eeg, axis=0)), axis=0) / len(eeg)
    else:
        gfp = np.sum(np.abs(eeg - np.median(eeg, axis=0)), axis=0) / len(eeg)
    return gfp

def _eeg_gfp_L2(eeg, robust=False):
    if robust is False:
        gfp = np.std(eeg, axis=0, ddof=0)
    else:
        gfp = mad(eeg, axis=0)
    return gfp




