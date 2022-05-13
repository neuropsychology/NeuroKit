# -*- coding: utf-8 -*-
import numpy as np

from ..signal import signal_filter


def emg_amplitude(emg_cleaned):
    """**Compute electromyography (EMG) amplitude**

    Compute electromyography amplitude given the cleaned respiration signal, done by calculating the
    linear envelope of the signal.

    Parameters
    ----------
    emg_cleaned : Union[list, np.array, pd.Series]
        The cleaned electromyography channel as returned by ``emg_clean()``.

    Returns
    -------
    array
        A vector containing the electromyography amplitude.

    See Also
    --------
    emg_clean, emg_rate, emg_process, emg_plot

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk
      import pandas as pd

      emg = nk.emg_simulate(duration=10, sampling_rate=1000, burst_number=3)
      cleaned = nk.emg_clean(emg, sampling_rate=1000)

      amplitude = nk.emg_amplitude(cleaned)
      @savefig p_emg_amplitude1.png scale=100%
      fig = pd.DataFrame({"EMG": emg, "Amplitude": amplitude}).plot(subplots=True)
      @suppress
      plt.close()

    """
    tkeo = _emg_amplitude_tkeo(emg_cleaned)
    amplitude = _emg_amplitude_envelope(tkeo)

    return amplitude


# =============================================================================
# Taeger-Kaiser Energy Operator
# =============================================================================
def _emg_amplitude_tkeo(emg_cleaned):
    """Calculates the Teager–Kaiser Energy operator to improve onset detection, described by Marcos Duarte at
    https://github.com/demotu/BMC/blob/master/notebooks/Electromyography.ipynb.

    Parameters
    ----------
    emg_cleaned : Union[list, np.array, pd.Series]
        The cleaned electromyography channel as returned by `emg_clean()`.

    Returns
    -------
    tkeo : array
        The emg signal processed by the Teager–Kaiser Energy operator.

    References
    ----------
    - BMCLab: https://github.com/demotu/BMC/blob/master/notebooks/Electromyography.ipynb

    - Li, X., Zhou, P., & Aruin, A. S. (2007). Teager–Kaiser energy operation of surface EMG improves
    muscle activity onset detection. Annals of biomedical engineering, 35(9), 1532-1538.

    """
    tkeo = emg_cleaned.copy()

    # Teager–Kaiser Energy operator
    tkeo[1:-1] = emg_cleaned[1:-1] * emg_cleaned[1:-1] - emg_cleaned[:-2] * emg_cleaned[2:]

    # Correct the data in the extremities
    tkeo[0], tkeo[-1] = tkeo[1], tkeo[-2]

    return tkeo


# =============================================================================
# Linear Envelope
# =============================================================================
def _emg_amplitude_envelope(
    emg_cleaned, sampling_rate=1000, lowcut=10, highcut=400, envelope_filter=8
):
    """Calculate the linear envelope of a signal.

    This function implements a 2nd-order Butterworth filter with zero lag, described by Marcos Duarte
    at <https://github.com/demotu/BMC/blob/master/notebooks/Electromyography.ipynb>.

    Parameters
    ----------
    emg_cleaned : Union[list, np.array, pd.Series]
        The cleaned electromyography channel as returned by `emg_clean()`.
    sampling_rate : int
        The sampling frequency of `emg_signal` (in Hz, i.e., samples/second).
    lowcut : float
        Low-cut frequency for the band-pass filter (in Hz). Defaults to 10Hz.
    highcut : float
        High-cut frequency for the band-pass filter (in Hz). Defaults to 400Hz.
    envelope_filter : float
        Cuttoff frequency for the high-pass filter (in Hz). Defauts to 8Hz.

    Returns
    -------
    envelope : array
        The linear envelope of the emg signal.

    References
    ----------
    - BMCLab: https://github.com/demotu/BMC/blob/master/notebooks/Electromyography.ipynb

    """
    filtered = signal_filter(
        emg_cleaned,
        sampling_rate=sampling_rate,
        lowcut=lowcut,
        highcut=highcut,
        method="butterworth",
        order=2,
    )

    envelope = np.abs(filtered)
    envelope = signal_filter(
        envelope,
        sampling_rate=sampling_rate,
        lowcut=None,
        highcut=envelope_filter,
        method="butterworth",
        order=2,
    )

    return envelope
