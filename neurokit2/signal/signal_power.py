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
    >>> signal = nk.signal_simulate(frequency=5) + 0.5*nk.signal_simulate(frequency=20)
    >>> nk.signal_power(signal, frequency_band=[(18, 22), (10, 14)], method="multitapers")
    """

    if continuous is False:
        out = _signal_power_instant(signal, frequency_band, sampling_rate=sampling_rate, **kwargs)
    else:
        raise ValueError("not implemented yet.")

#    freqs = freq_bands[band]
#    # Filter to keep only the band of interest
#    filtered, sampling_rate, params = biosppy.signals.tools.filter_signal(signal=RRi, ftype='butter', band='bandpass', order=1, frequency=freqs, sampling_rate=sampling_rate)
#    # Apply Hilbert transform
#    amplitude, phase = biosppy.signals.tools.analytic_signal(filtered)

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
    out[str(frequency_band[0]) + "-" + str(frequency_band[1]) + "Hz"] = np.trapz(y=psd["Power"][indices], x=psd["Frequency"][indices])
    return out
