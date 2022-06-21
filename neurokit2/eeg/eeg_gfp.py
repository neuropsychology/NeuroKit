# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal import signal_filter
from ..stats import mad, standardize


def eeg_gfp(
    eeg,
    sampling_rate=None,
    method="l1",
    normalize=False,
    smooth=0,
    robust=False,
    standardize_eeg=False,
):
    """**Global Field Power (GFP)**

    Global Field Power (GFP) constitutes a reference-independent measure of response strength.
    GFP was first introduced by Lehmann and Skrandies (1980) and has since become a commonplace
    measure among M/EEG users. Mathematically, GFP is the standard deviation of all electrodes
    at a given time.

    Parameters
    ----------
    eeg : array
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second). Only necessary if
        smoothing is requested.
    method : str
        Can be either ``l1`` or ``l2`` to use the L1 or L2 norm.
    normalize : bool
        Normalize GFP.
    smooth : float
        Can be either ``None`` or a float. If a float, will use this value, multiplied by the
        sampling rate.
    robust : bool
        If ``True``, the GFP extraction (and the data standardization if requested) will be done
        using the median/MAD instead of the mean/SD.
    standardize_eeg : bool
        Standardize (z-score) the data across time prior to GFP extraction using ``nk.standardize()``.

    Returns
    -------
    gfp : array
        The global field power of each sample point in the data.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      eeg = nk.mne_data("filt-0-40_raw")
      eeg = nk.eeg_rereference(eeg, 'average')
      eeg = eeg.get_data()[:, 0:500]  # Get the 500 first data points

    * **Example 1:** Compare L1 and L2 norms

    .. ipython:: python

      l1 = nk.eeg_gfp(eeg, method="l1", normalize=True)
      l2 = nk.eeg_gfp(eeg, method="l2", normalize=True)
      @savefig p_eeg_gfp1.png scale=100%
      nk.signal_plot([l1, l2])
      @suppress
      plt.close()

    * **Example 2:** Compare Mean-based and Median-based

    .. ipython:: python

      gfp = nk.eeg_gfp(eeg, normalize=True)
      gfp_r = nk.eeg_gfp(eeg, normalize=True, robust=True)
      @savefig p_eeg_gfp2.png scale=100%
      nk.signal_plot([gfp, gfp_r])
      @suppress
      plt.close()

    * **Example 3:** Standardize the data

    .. ipython:: python

      gfp = nk.eeg_gfp(eeg, normalize=True)
      gfp_z = nk.eeg_gfp(eeg, normalize=True, standardize_eeg=True)
      @savefig p_eeg_gfp3.png scale=100%
      nk.signal_plot([gfp, gfp_z])
      @suppress
      plt.close()

    References
    ----------
    * Lehmann, D., & Skrandies, W. (1980). Reference-free identification of components of
      checkerboard-evoked multichannel potential fields. Electroencephalography and clinical
      neurophysiology, 48(6), 609-621.

    """
    # If MNE object
    if isinstance(eeg, (pd.DataFrame, np.ndarray)) is False:
        sampling_rate = eeg.info["sfreq"]
        eeg = eeg.get_data()

    # Normalization
    if standardize_eeg is True:
        eeg = standardize(eeg, robust=robust)

    # Compute GFP
    if method.lower() == "l1":
        gfp = _eeg_gfp_L1(eeg, robust=robust)
    else:
        gfp = _eeg_gfp_L2(eeg, robust=robust)

    # Normalize (between 0 and 1)
    if normalize is True:
        gfp = gfp / np.max(gfp)

    # Smooth
    if smooth is not None and smooth != 0:
        gfp = _eeg_gfp_smoothing(gfp, sampling_rate=sampling_rate, window_size=smooth)

    return gfp


# =============================================================================
# Utilities
# =============================================================================
def _eeg_gfp_smoothing(gfp, sampling_rate=None, window_size=0.02):
    """Smooth the Global Field Power Curve"""
    if sampling_rate is None:
        raise ValueError(
            "NeuroKit error: eeg_gfp(): You requested to smooth the GFP, for which ",
            "we need to know the sampling_rate. Please provide it as an argument.",
        )
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
