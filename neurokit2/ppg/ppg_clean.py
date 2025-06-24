# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np

from ..misc import NeuroKitWarning, as_vector
from ..signal import signal_fillmissing, signal_filter


def ppg_clean(ppg_signal, sampling_rate=1000, heart_rate=None, method="elgendi"):
    """**Clean a photoplethysmogram (PPG) signal**

    Prepare a raw PPG signal for systolic peak detection.

    Parameters
    ----------
    ppg_signal : Union[list, np.array, pd.Series]
        The raw PPG channel.
    heart_rate : Union[int, float]
        The heart rate of the PPG signal. Applicable only if method is ``"nabian2018"`` to check
        that filter frequency is appropriate.
    sampling_rate : int
        The sampling frequency of the PPG (in Hz, i.e., samples/second). The default is 1000.
    method : str
        The processing pipeline to apply. Can be one of ``"elgendi"``, ``"nabian2018"``, or ``"none"``.
        The default is ``"elgendi"``. If ``"none"`` is passed, the raw signal will be returned without
        any cleaning.

    Returns
    -------
    clean : array
        A vector containing the cleaned PPG.

    See Also
    --------
    ppg_simulate, ppg_findpeaks

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk
      import pandas as pd
      import matplotlib.pyplot as plt

      # Simulate and clean signal
      ppg = nk.ppg_simulate(heart_rate=75, duration=30)
      ppg_elgendi = nk.ppg_clean(ppg, method='elgendi')
      ppg_nabian = nk.ppg_clean(ppg, method='nabian2018', heart_rate=75)

      # Plot and compare methods
      signals = pd.DataFrame({'PPG_Raw' : ppg,
                              'PPG_Elgendi' : ppg_elgendi,
                              'PPG_Nabian' : ppg_nabian})
      @savefig p_ppg_clean1.png scale=100%
      signals.plot()
      @suppress
      plt.close()


    References
    ----------
    * Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., & Ostadabbas, S. (2018).
      An open-source feature extraction tool for the analysis of peripheral physiological data.
      IEEE Journal of Translational Engineering in Health and Medicine, 6, 1-11.
    * M. Elgendi, I. Norton, M. Brearley, D. Abbott, and D. Schuurmans (2013).
      Systolic peak detection in acceleration photoplethysmograms measured from emergency responders
      in tropical conditions. PLoS ONE, 8(10), 1–11.

    """
    ppg_signal = as_vector(ppg_signal)

    # Missing data
    n_missing = np.sum(np.isnan(ppg_signal))
    if n_missing > 0:
        warn(
            "There are " + str(n_missing) + " missing data points in your signal."
            " Filling missing values using `signal_fillmissing`.",
            category=NeuroKitWarning,
        )
        ppg_signal = signal_fillmissing(ppg_signal, method="both")

    method = str(method).lower()
    if method in ["elgendi", "elgendi2013"]:
        clean = _ppg_clean_elgendi(ppg_signal, sampling_rate)
    elif method in ["nabian2018"]:
        clean = _ppg_clean_nabian2018(ppg_signal, sampling_rate, heart_rate=heart_rate)
    elif method in ["none"]:
        clean = ppg_signal
    else:
        raise ValueError(
            "`method` not found. Must be one of 'elgendi', 'nabian2018', or 'none'."
        )

    return clean


# =============================================================================
# Methods
# =============================================================================


def _ppg_clean_elgendi(ppg_signal, sampling_rate):
    """Low-pass filter for continuous PPG signal preprocessing, adapted from Elgendi et al. (2013)."""
    filtered = signal_filter(
        ppg_signal,
        sampling_rate=sampling_rate,
        lowcut=0.5,
        highcut=8,
        order=2,
        method="butterworth",
    )
    return filtered


def _ppg_clean_nabian2018(ppg_signal, sampling_rate, heart_rate=None):
    """Low-pass filter for continuous BP signal preprocessing, adapted from Nabian et al. (2018)."""

    # Determine low-pass filter value
    highcut = 40

    # Convert heart rate to seconds, check if low-pass filter within appropriate range
    if heart_rate is not None:
        heart_rate = heart_rate / 60

        if not highcut >= 10 * heart_rate and not highcut < 0.5 * sampling_rate:
            raise ValueError(
                "Highcut value should be at least 10 times heart rate and"
                " less than 0.5 times sampling rate."
            )

    filtered = signal_filter(
        ppg_signal,
        sampling_rate=sampling_rate,
        lowcut=None,
        highcut=highcut,
        order=2,
        method="butterworth",
    )

    return filtered
