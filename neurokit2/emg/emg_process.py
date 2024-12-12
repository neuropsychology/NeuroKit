# -*- coding: utf-8 -*-
import pandas as pd

from ..misc.report import create_report
from ..signal import signal_sanitize
from .emg_activation import emg_activation
from .emg_amplitude import emg_amplitude
from .emg_clean import emg_clean
from .emg_methods import emg_methods
from .emg_plot import emg_plot


def emg_process(emg_signal, sampling_rate=1000, report=None, **kwargs):
    """**Process a electromyography (EMG) signal**

    Convenience function that automatically processes an electromyography signal.

    Parameters
    ----------
    emg_signal : Union[list, np.array, pd.Series]
        The raw electromyography channel.
    sampling_rate : int
        The sampling frequency of ``emg_signal`` (in Hz, i.e., samples/second).
    report : str
        The filename of a report containing description and figures of processing
        (e.g. ``"myreport.html"``). Needs to be supplied if a report file
        should be generated. Defaults to ``None``. Can also be ``"text"`` to
        just print the text in the console without saving anything.
    **kwargs
        Other arguments to be passed to specific methods. For more information,
        see :func:`.emg_methods`.

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as ``emg_signal`` containing the following columns:

        .. codebookadd::
            EMG_Raw|The raw EMG signal.
            EMG_Clean|The cleaned EMG signal.
            EMG_Amplitude|The signal amplitude, or the activation of the signal.
            EMG_Activity|The activity of the signal for which amplitude exceeds the threshold \
                specified,marked as "1" in a list of zeros.
            EMG_Onsets|The onsets of the amplitude, marked as "1" in a list of zeros.
            EMG_Offsets|The offsets of the amplitude, marked as "1" in a list of zeros.

    info : dict
        A dictionary containing the information of each amplitude onset, offset, and peak activity
        (see :func:`emg_activation`), as well as the signals' sampling rate.

    See Also
    --------
    emg_clean, emg_amplitude, emg_plot

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      emg = nk.emg_simulate(duration=10, sampling_rate=1000, burst_number=3)
      signals, info = nk.emg_process(emg, sampling_rate=1000)

      @savefig p_emg_process1.png scale=100%
      nk.emg_plot(signals, info)
      @suppress
      plt.close()

    """
    # Sanitize input
    emg_signal = signal_sanitize(emg_signal)
    methods = emg_methods(sampling_rate=sampling_rate, **kwargs)

    # Clean signal
    emg_cleaned = emg_clean(
        emg_signal, sampling_rate=sampling_rate, method=methods["method_cleaning"]
    )

    # Get amplitude
    amplitude = emg_amplitude(emg_cleaned)

    # Get onsets, offsets, and periods of activity
    activity_signal, info = emg_activation(
        emg_amplitude=amplitude,
        emg_cleaned=emg_cleaned,
        sampling_rate=sampling_rate,
        method=methods["method_activation"],
        **methods["kwargs_activation"]
    )
    info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

    # Prepare output
    signals = pd.DataFrame(
        {"EMG_Raw": emg_signal, "EMG_Clean": emg_cleaned, "EMG_Amplitude": amplitude}
    )

    signals = pd.concat([signals, activity_signal], axis=1)

    if report is not None:
        # Generate report containing description and figures of processing
        if ".html" in str(report):
            fig = emg_plot(signals, info, static=False)
        else:
            fig = None
        create_report(file=report, signals=signals, info=methods, fig=fig)

    return signals, info
