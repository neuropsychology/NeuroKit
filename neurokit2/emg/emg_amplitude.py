# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal

def emg_amplitude(emg_cleaned):
    """Compute electromyography amplitude.

    Compute electromyography amplitude given the cleaned respiration signal, done by calculating the linear envelope of the signal.

    Parameters
    ----------
    emg_cleaned : list, array or Series
        The cleaned electromyography channel as returned by `emg_clean()`.

    Returns
    -------
    array
        A vector containing the electromyography amplitude.

    See Also
    --------
    emg_clean, emg_rate, emg_process, emg_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> emg = nk.emg_simulate(duration=10, sampling_rate=1000, n_bursts=3)
    >>> cleaned = nk.emg_clean(emg, sampling_rate=1000)
    >>> amplitude = nk.emg_amplitude(cleaned)
    >>> signal_plot(pd.DataFrame({"EMG": emg, "Amplitude": amplitude}), subplots=True)
    """
    emg = _emg_tkeo(emg_cleaned)
    emg = _emg_linear_envelope(emg_cleaned)

    return(emg)


# =============================================================================
# Taeger-Kaiser Energy Operator
# =============================================================================
def _emg_tkeo(emg_cleaned):
    """Calculates the Teager–Kaiser Energy operator to improve onset detection, described by Marcos Duarte at <https://github.com/demotu/BMC/blob/master/notebooks/Electromyography.ipynb>.

    Parameters
    ----------
    emg_cleaned : list, array or Series
        The cleaned electromyography channel as returned by `emg_clean()`.

    Returns
    -------
    tkeo : array
        The emg signal processed by the Teager–Kaiser Energy operator.

    References
    ----------
    - BMCLab: https://github.com/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
    - Li, X., Zhou, P., & Aruin, A. S. (2007). Teager–Kaiser energy operation of surface EMG improves muscle activity onset detection. Annals of biomedical engineering, 35(9), 1532-1538.
    """
    emg_cleaned = np.asarray(emg_cleaned)
    tkeo = np.copy(emg_cleaned)
    # Teager–Kaiser Energy operator
    tkeo[1:-1] = emg_cleaned[1:-1]*emg_cleaned[1:-1] - emg_cleaned[:-2]*emg_cleaned[2:]
    # correct the data in the extremities
    tkeo[0], tkeo[-1] = tkeo[1], tkeo[-2]

    return(tkeo)




# =============================================================================
# Linear Envelope
# =============================================================================
def _emg_linear_envelope(emg_cleaned, sampling_rate=1000, freqs=[10, 400], lfreq=8):
    """Calculate the linear envelope of a signal.

    This function implements a 2nd-order Butterworth filter with zero lag, described by Marcos Duarte at <https://github.com/demotu/BMC/blob/master/notebooks/Electromyography.ipynb>.

    Parameters
    ----------
    emg_cleaned : list, array or Series
        The cleaned electromyography channel as returned by `emg_clean()`.
    sampling_rate : int
        The sampling frequency of `emg_signal` (in Hz, i.e., samples/second).
    freqs : list [fc_h, fc_l], optional
            cutoff frequencies for the band-pass filter (in Hz). Defaults to [10, 400].
    lfreq : number, optional
            cutoff frequency for the low-pass filter (in Hz). Defaults to 8Hz.

    Returns
    -------
    envelope : array
        The linear envelope of the emg signal.

    References
    ----------
    - BMCLab: https://github.com/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
    """
    if np.size(freqs) == 2:
        # band-pass filter
        b, a = scipy.signal.butter(2, np.array(freqs)/(sampling_rate/2.), btype='bandpass')
        emg_signal = scipy.signal.filtfilt(b, a, emg_cleaned)
    if np.size(lfreq) == 1:
        # full-wave rectification
        envelope = abs(emg_cleaned)
        # low-pass Butterworth filter
        b, a = scipy.signal.butter(2, np.array(lfreq)/(sampling_rate/2.), btype='low')
        envelope = scipy.signal.filtfilt(b, a, envelope)



    return (envelope)
