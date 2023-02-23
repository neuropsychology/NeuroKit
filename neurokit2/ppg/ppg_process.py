# -*- coding: utf-8 -*-
import pandas as pd

from ..misc import as_vector
from ..misc.report import create_report
from ..signal import signal_rate
from ..signal.signal_formatpeaks import _signal_from_indices
from .ppg_clean import ppg_clean
from .ppg_findpeaks import ppg_findpeaks
from .ppg_methods import ppg_methods
from .ppg_plot import ppg_plot

def ppg_process(ppg_signal, sampling_rate=1000, method="elgendi", report=None, **kwargs):
    """**Process a photoplethysmogram (PPG)  signal**

    Convenience function that automatically processes a photoplethysmogram signal.

    Parameters
    ----------
    ppg_signal : Union[list, np.array, pd.Series]
        The raw PPG channel.
    sampling_rate : int
        The sampling frequency of :func:`.ppg_signal` (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of ``"elgendi"``.
        Defaults to ``"elgendi"``.
    report : str
        The filename of a report containing description and figures of processing
        (e.g. ``"myreport.html"``). Needs to be supplied if a report file
        should be generated. Defaults to ``None``. Can also be ``"text"`` to
        just print the text in the console without saving anything.
    **kwargs
        Other arguments to be passed to specific methods. For more information,
        see :func:`.ppg_methods`.

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as :func:`.ppg_signal` containing the following columns:

        * ``"PPG_Raw"``: the raw signal.
        * ``"PPG_Clean"``: the cleaned signal.
        * ``"PPG_Rate"``: the heart rate as measured based on PPG peaks.
        * ``"PPG_Peaks"``: the PPG peaks marked as "1" in a list of zeros.

    info : dict
        A dictionary containing the information of peaks and the signals' sampling rate.

    See Also
    --------
    ppg_clean, ppg_findpeaks

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      ppg = nk.ppg_simulate(duration=10, sampling_rate=1000, heart_rate=70)
      signals, info = nk.ppg_process(ppg, sampling_rate=1000)
      @savefig p_ppg_process1.png scale=100%
      nk.ppg_plot(signals)
      @suppress
      plt.close()

    """
    # Sanitize input
    ppg_signal = as_vector(ppg_signal)
    methods = ppg_methods(sampling_rate=sampling_rate, method=method, **kwargs)

    if methods["method_cleaning"] is None or methods["method_cleaning"].lower() == "none":
        ppg_cleaned = ppg_signal
    else:
        # Clean signal
        ppg_cleaned = ppg_clean(
            ppg_signal,
            sampling_rate=sampling_rate,
            method=methods["method_cleaning"],
            **methods["kwargs_cleaning"]
        )

    # Find peaks
    info = ppg_findpeaks(
        ppg_cleaned,
        sampling_rate=sampling_rate,
        method=methods["method_peaks"],
        **methods["kwargs_peaks"]
    )

    info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

    # Mark peaks
    peaks_signal = _signal_from_indices(info["PPG_Peaks"], desired_length=len(ppg_cleaned))

    # Rate computation
    rate = signal_rate(
        info["PPG_Peaks"], sampling_rate=sampling_rate, desired_length=len(ppg_cleaned)
    )

    # Prepare output
    signals = pd.DataFrame(
        {
            "PPG_Raw": ppg_signal,
            "PPG_Clean": ppg_cleaned,
            "PPG_Rate": rate,
            "PPG_Peaks": peaks_signal,
        }
    )

    if report is not None:
        # Generate report containing description and figures of processing
        if ".html" in report:
            fig = ppg_plot(signals, sampling_rate=sampling_rate, static=False)
        else:
            fig = None
        create_report(file=report, signals=signals, info=methods, fig=fig)

    return signals, info
