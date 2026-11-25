import pandas as pd

from ..signal import signal_power
from .utils import _sanitize_eeg


def eeg_power(eeg, sampling_rate=None, frequency_band=None, **kwargs):
    """**EEG Power in Different Frequency Bands**

    See our `walkthrough <https://neuropsychology.github.io/NeuroKit/examples/eeg_power/eeg_power.html>`_ for
    details.

    * **Gamma** (30-80 Hz)
    * **Beta** (13-30 Hz)
      * **Beta 1** (13-16 Hz)
      * **Beta 2** (16-20 Hz)
      * **Beta 3** (20-30 Hz)
    * **SMR** (13-15 Hz)
    * **Alpha** (8-13 Hz)
    * **Mu** (9-11 Hz)
    * **Theta** (4-8 Hz)
    * **Delta** (1-4 Hz)

    Parameters
    ----------
    eeg : array
        An array (channels, times) of M/EEG data or a Raw or Epochs object from MNE.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second). Only necessary if
        smoothing is requested.
    frequency_band : list
        A list of frequency bands (or tuples of frequencies).
    **kwargs
        Other arguments to be passed to ``nk.signal_power()``.

    Returns
    -------
    pd.DataFrame
        The power in different frequency bands for each channel.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      # Raw objects
      eeg = nk.mne_data("raw")
      by_channel = nk.eeg_power(eeg)
      by_channel.head()

      average = by_channel.mean(numeric_only=True, axis=0)
      average["Gamma"]

    References
    ----------
    - Lehmann, D., & Skrandies, W. (1980). Reference-free identification of components of
      checkerboard-evoked multichannel potential fields. Electroencephalography and clinical
      neurophysiology, 48(6), 609-621.

    """

    # Sanitize names and values
    frequency_band, band_names = _eeg_power_sanitize(frequency_band)

    # Sanitize input
    eeg, sampling_rate, _ = _sanitize_eeg(eeg, sampling_rate=sampling_rate)
    if sampling_rate is None:
        raise ValueError("NeuroKit error: eeg_power(): 'sampling_rate' must be provided when 'eeg' is an array or DataFrame.")

    # Iterate through channels
    data = []
    for channel in eeg.columns:
        rez = signal_power(
            eeg[channel].values,
            sampling_rate=sampling_rate,
            frequency_band=frequency_band,
            **kwargs,
        )
        data.append(rez)

    data = pd.concat(data, axis=0, ignore_index=True)
    data.columns = band_names
    data.insert(0, "Channel", eeg.columns)

    return data


# =============================================================================
# Utilities
# =============================================================================
def _eeg_power_sanitize(frequency_band=None):
    if frequency_band is None:
        frequency_band = ["Gamma", "Beta", "Alpha", "Theta", "Delta"]
    elif isinstance(frequency_band, (str, tuple)):
        frequency_band = [frequency_band]

    band_ranges = {
        "gamma": (30, 80),
        "beta": (13, 30),
        "beta1": (13, 16),
        "beta2": (16, 20),
        "beta3": (20, 30),
        "smr": (13, 15),
        "alpha": (8, 13),
        "mu": (9, 11),
        "theta": (4, 8),
        "delta": (1, 4),
    }

    bands_out = []
    band_names = []
    for f in frequency_band:
        if isinstance(f, str):
            f_key = f.lower().replace(" ", "").replace("_", "")
            if f_key not in band_ranges:
                raise ValueError(f"Unknown frequency band: '{f.lower()}'")
            bands_out.append(band_ranges[f_key])
            band_names.append(f)
        elif isinstance(f, (tuple, list)) and len(f) == 2:
            bands_out.append((f[0], f[1]))
            band_names.append(f"Hz_{f[0]}_{f[1]}")
        else:
            raise ValueError("'frequency_band' must be a list of tuples (or strings).")
    return bands_out, band_names
