# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal

from .signal_psd import signal_psd


def signal_power(signal, frequency_band, sampling_rate=1000, continuous=False, **kwargs):
    """Compute the power of a signal in a given frequency band.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    frequency_band :tuple or list
        Tuple or list of tuples indicating the range of frequencies to compute the power in.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    continuous : bool
        Compute instant frequency, or continuous power.
    show : bool
        If True, will return a Poincaré plot. Defaults to False.
    **kwargs
        Keyword arguments to be passed to `signal_psd()`.

    See Also
    --------
    signal_filter, signal_psd

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Power Spectrum values and a plot if
        `show` is True.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Instant power
    >>> signal = nk.signal_simulate(frequency=5) + 0.5*nk.signal_simulate(frequency=20)
    >>> nk.signal_power(signal, frequency_band=[(18, 22), (10, 14)], method="multitapers")
    >>>
    >>> # Continuous (simulated signal)
    >>> signal = np.concatenate((nk.ecg_simulate(duration=30, heart_rate=75),
                                 nk.ecg_simulate(duration=30, heart_rate=85)))
    >>> power = nk.signal_power(signal, frequency_band=[(72/60, 78/60), (82/60, 88/60)], continuous=True)
    >>> processed, _ = nk.ecg_process(signal)
    >>> power["ECG_Rate"] = processed["ECG_Rate"]
    >>> nk.signal_plot(power, standardize=True)
    >>>
    >>> # Continuous (real signal)
    >>> signal = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/dev/data/bio_eventrelated_100hz.csv")["ECG"]
    >>> power = nk.signal_power(signal, sampling_rate=100, frequency_band=[(0.12, 0.15), (0.15, 0.4)], continuous=True)
    >>> processed, _ = nk.ecg_process(signal, sampling_rate=100)
    >>> power["ECG_Rate"] = processed["ECG_Rate"]
    >>> nk.signal_plot(power, standardize=True)
    """

    if continuous is False:
        out = _signal_power_instant(signal, frequency_band, sampling_rate=sampling_rate, **kwargs)
    else:
        out = _signal_power_continuous(signal, frequency_band, sampling_rate=sampling_rate)

    out = pd.DataFrame.from_dict(out, orient="index").T
    return out



# =============================================================================
# Instant
# =============================================================================

def _signal_power_instant(signal, frequency_band, sampling_rate=1000, **kwargs):
    psd = signal_psd(signal, sampling_rate=sampling_rate, show=False, **kwargs)

    out = {}
    if isinstance(frequency_band[0], list) or isinstance(frequency_band[0], tuple):
        for band in frequency_band:
            out.update(_signal_power_instant_get(psd, band))
    else:
        out.update(_signal_power_instant_get(psd, frequency_band))
    return out



def _signal_power_instant_get(psd, frequency_band):

    indices = np.logical_and(psd["Frequency"] >= frequency_band[0], psd["Frequency"] < frequency_band[1]).values

    out = {}
    out["{:.2f}-{:.2f}Hz".format(frequency_band[0], frequency_band[1])] = np.trapz(y=psd["Power"][indices], x=psd["Frequency"][indices])
    return out

# =============================================================================
# Continuous
# =============================================================================

def _signal_power_continuous(signal, frequency_band, sampling_rate=1000):

    out = {}
    if isinstance(frequency_band[0], list) or isinstance(frequency_band[0], tuple):
        for band in frequency_band:
            out.update(_signal_power_continuous_get(signal, band, sampling_rate))
    else:
        out.update(_signal_power_continuous_get(signal, frequency_band, sampling_rate))
    return out



def _signal_power_continuous_get(signal, frequency_band, sampling_rate=1000, precision=20):

    try:
        import mne
    except ImportError:
        raise ImportError("NeuroKit warning: signal_power(): the 'mne'",
                          "module is required. ",
                          "Please install it first (`pip install mne`).")

    out = mne.time_frequency.tfr_array_morlet([[signal]],
                                              sfreq=sampling_rate,
                                              freqs=np.linspace(frequency_band[0], frequency_band[1], precision),
                                              output='power')
    power = np.mean(out[0][0], axis=0)

    out = {}
    out["{:.2f}-{:.2f}Hz".format(frequency_band[0], frequency_band[1])] = power
    return out

# def _signal_power_continuous_plot(signal, frequency_band, sampling_rate=1000):
    # if frequency_band=

    # frequency_band=[(0.12, 0.15), (0.15, 0.4)],

    # ulf=(0, 0.0033), vlf=(0.0033, 0.04), lf=(0.04, 0.15), hf=(0.15, 0.4), vhf=(0.4, 0.5)
