# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .mne_to_df import mne_to_df


def eeg_power(eeg, sampling_rate=None, frequency_band=["gamma", "beta", "alpha", "theta", "delta"]):
    """EEG Power in Different Frequency Bands

    - Gamma (30-40 Hz)
    - Beta (13-30 Hz)
    - Alpha (8-13 Hz)
    - Theta (4-8 Hz)
    - Delta (1-4 Hz)

    Parameters
    ----------
    eeg : array
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second). Only necessary if
        smoothing is requested.

    Returns
    -------
    gfp : array
        The global field power of each sample point in the data.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> eeg = nk.mne_data("filt-0-40_raw")
    >>> eeg = eeg.get_data()[:, 0:500]  # Get the 500 first data points
    >>>
    >>> frequency_band=["gamma", (1, 4), "Alpha"]

    References
    ----------
    - Lehmann, D., & Skrandies, W. (1980). Reference-free identification of components of
    checkerboard-evoked multichannel potential fields. Electroencephalography and clinical
    neurophysiology, 48(6), 609-621.

    """

    # Sanitize names and values
    bands = frequency_band.copy()  # This will used for the names
    for i, f in enumerate(frequency_band):
        if isinstance(f, str):
            f_name = f.lower()
            if f_name == "gamma":
                frequency_band[i] = (30, 40)
            elif f_name == "beta":
                frequency_band[i] = (13, 30)
            elif f_name == "alpha":
                frequency_band[i] = (8, 13)
            elif f_name == "theta":
                frequency_band[i] = (4, 8)
            elif f_name == "delta":
                frequency_band[i] = (1, 4)
            else:
                raise ValueError(f"Unknown frequency band: '{f_name}'")
        if isinstance(f, tuple):
            bands[i] = "Freq_" + str(f[0]) + "_" + str(f[1])
        else:
            raise ValueError("'frequency_band' must be a list of tuples (or strings).")

    # Sanitize input
    # If MNE object
    if isinstance(eeg, (pd.DataFrame, np.ndarray)) is False:
        sampling_rate = eeg.info["sfreq"]
        eeg = mne_to_df(eeg)
        eeg.values.dim()

    pass
