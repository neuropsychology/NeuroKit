# -*- coding: utf-8 -*-
import pandas as pd

from .rsp_clean import rsp_clean
from .rsp_phase import rsp_phase
from .rsp_peaks import rsp_peaks
from .rsp_rate import rsp_rate
from .rsp_amplitude import rsp_amplitude



def rsp_process(rsp_signal, sampling_rate=1000, method="khodadad2018"):
    """Process a respiration (RSP) signal.

    Convenience function that automatically processes a respiration signal with
    one of the following methods:

    - `Khodadad et al. (2018)
    <https://iopscience.iop.org/article/10.1088/1361-6579/aad7e6/meta>`_
    - `BioSPPy
    <https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/resp.py>`_

    Parameters
    ----------
    rsp_signal : list, array or Series
        The raw respiration channel (as measured, for instance, by a
        respiration belt).
    sampling_rate : int
        The sampling frequency of `rsp_signal` (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of "khodadad2018"
        (default) or "biosppy".

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as `rsp_signal` containing the following
        columns:

        - *"RSP_Raw"*: the raw signal.
        - *"RSP_Clean"*: the cleaned signal.
        - *"RSP_Peaks"*: the inhalation peaks marked as "1" in a list of zeros.
        - *"RSP_Troughs"*: the exhalation troughs marked as "1" in a list of
                            zeros.
        - *"RSP_Rate"*: breathing rate interpolated between inhalation peaks.
        - *"RSP_Amplitude"*: breathing amplitude interpolated between
                                inhalation peaks.
    info : dict
        A dictionary containing the samples at which inhalation peaks and
        exhalation troughs occur, accessible with the keys "RSP_Peaks", and
        "RSP_Troughs", respectively.

    See Also
    --------
    rsp_clean, rsp_findpeaks, rsp_rate, rsp_amplitude, rsp_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(duration=90, respiratory_rate=15)
    >>> signals, info = nk.rsp_process(rsp, sampling_rate=1000)
    >>> nk.rsp_plot(signals)
    """
    # Clean signal
    rsp_cleaned = rsp_clean(rsp_signal, sampling_rate=sampling_rate,
                            method=method)

    # Extract, fix and format peaks
    info, peak_signal = rsp_peaks(rsp_cleaned,
                                  sampling_rate=sampling_rate,
                                  method=method,
                                  amplitude_min=0.3)

    # Get additional parameters
    phase = rsp_phase(peak_signal)
    amplitude = rsp_amplitude(rsp_cleaned, peak_signal)
    rate = rsp_rate(peak_signal, sampling_rate=sampling_rate, method=method)

    # Prepare output
    signals = pd.DataFrame({"RSP_Raw": rsp_signal,
                            "RSP_Clean": rsp_cleaned,
                            "RSP_Inspiration": phase,
                            "RSP_Amplitude": amplitude,
                            "RSP_Rate": rate})
    signals = pd.concat([signals, peak_signal], axis=1)

    return signals, info
