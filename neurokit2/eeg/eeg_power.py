# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal import signal_power
from .mne_to_df import mne_to_df


def eeg_power(
    eeg, sampling_rate=None, frequency_band=["Gamma", "Beta", "Alpha", "Theta", "Delta"], **kwargs
):
    """EEG Power in Different Frequency Bands

    See our `walkthrough <https://neurokit2.readthedocs.io/en/latest/examples/eeg_power.html>`_ for details.

    - **Gamma** (30-80 Hz)
    - **Beta** (13-30 Hz)
      - **Beta 1** (13–16 Hz)
      - **Beta 2** (16–20 Hz)
      - **Beta 3** (20–30 Hz)
    - **SMR** (13-15 Hz)
    - **Alpha** (8-13 Hz)
    - **Mu** (9-11 Hz)
    - **Theta** (4-8 Hz)
    - **Delta** (1-4 Hz)

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
    >>> # Raw objects
    >>> eeg = nk.mne_data("raw")
    >>> by_channel = nk.eeg_power(eeg)
    >>> by_channel #doctest: +SKIP

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
                frequency_band[i] = (30, 80)
            elif f_name == "beta":
                frequency_band[i] = (13, 30)
            elif f_name == "beta1":
                frequency_band[i] = (13, 16)
            elif f_name == "beta2":
                frequency_band[i] = (16, 20)
            elif f_name == "beta3":
                frequency_band[i] = (20, 30)
            elif f_name == "smr":
                frequency_band[i] = (13, 15)
            elif f_name == "alpha":
                frequency_band[i] = (8, 13)
            elif f_name == "mu":
                frequency_band[i] = (9, 11)
            elif f_name == "theta":
                frequency_band[i] = (4, 8)
            elif f_name == "delta":
                frequency_band[i] = (1, 4)
            else:
                raise ValueError(f"Unknown frequency band: '{f_name}'")
        elif isinstance(f, tuple):
            bands[i] = f"Hz_{f[0]}_{f[1]}"
        else:
            raise ValueError("'frequency_band' must be a list of tuples (or strings).")

    # Sanitize input
    if isinstance(eeg, np.ndarray):
        eeg = pd.DataFrame(eeg.T)  # input should be (channels, times)

    # Probably an mne object
    if isinstance(eeg, pd.DataFrame) is False:
        sampling_rate = eeg.info["sfreq"]
        eeg = mne_to_df(eeg)
        eeg = eeg.drop(columns=["Time"])

    data = []
    for channel in eeg.columns:
        rez = signal_power(
            eeg[channel].values,
            sampling_rate=sampling_rate,
            frequency_band=frequency_band,
            **kwargs,
        )
        data.append(rez)

    data = pd.concat(data, axis=0)
    data.columns = bands
    data.insert(0, "Channel", eeg.columns)
    data.reset_index(drop=True, inplace=True)

    return data
