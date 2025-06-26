# - * - coding: utf-8 - * -
import numpy as np

from ..epochs import epochs_create
from ..signal.signal_rate import signal_rate

def signal_cyclesegment(signal_cleaned, cycle_indices, sampling_rate=1000, **kwargs):
    """**Segment a signal into individual cycles**

    Segment a signal (e.g. ECG, PPG, respiratory) into individual cycles (e.g. heartbeats, pulse waves, breaths).

    Parameters
    ----------
    signal_cleaned : Union[list, np.array, pd.Series]
        The cleaned signal channel, such as that returned by ``ppg_clean()`` or ``ecg_clean()``.
    cycle_indices : dict
        The samples indicating individual cycles (such as PPG peaks or ECG R-peaks), such as a dict
        returned by ``ppg_peaks()``.
    sampling_rate : int
        The sampling frequency of ``signal_cleaned`` (in Hz, i.e., samples/second). Defaults to 1000.
    **kwargs
        Other arguments to be passed.

    Returns
    -------
    dict
        A dict containing DataFrames for all segmented cycles.

    See Also
    --------
    ppg_clean, ecg_clean, ppg_peaks, ecg_peaks, ppg_quality, ecg_quality

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk
      
      sampling_rate = 100
      ppg = nk.ppg_simulate(duration=30, sampling_rate=sampling_rate, heart_rate=80)
      ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=sampling_rate)
      _, peaks = nk.ppg_peaks(ppg_cleaned, sampling_rate=sampling_rate)
      peaks = peaks["PPG_Peaks"]
      heartbeats = nk.signal_cyclesegment(ppg_cleaned, peaks, sampling_rate=sampling_rate)

    """

    # To-do: This doesn't currently contain the plotting functionality of ecg_segment.

    if len(signal_cleaned) < sampling_rate * 4:
        raise ValueError("The data length is too small to be segmented.")

    epochs_start, epochs_end, average_cycle_rate = _segment_window(
        cycle_indices=cycle_indices,
        sampling_rate=sampling_rate,
        desired_length=len(signal_cleaned),
        ratio_pre=0.5,
    )
    cycles = epochs_create(
        signal_cleaned,
        cycle_indices,
        sampling_rate=sampling_rate,
        epochs_start=epochs_start,
        epochs_end=epochs_end,
    )

    # pad last cycle with nan so that segments are equal length
    last_cycle_key = str(np.max(np.array(list(cycles.keys()), dtype=int)))
    outside_bounds = cycles[last_cycle_key]["Index"] >= len(signal_cleaned)
    cycles[last_cycle_key].loc[outside_bounds, "Signal"] = np.nan

    return cycles


def _segment_window(
    cycle_rate=None,
    cycle_indices=None,
    sampling_rate=1000,
    desired_length=None,
    ratio_pre=0.5,
):
    # Extract cycle rate
    if cycle_rate is not None:
        cycle_rate = np.mean(cycle_rate)
    if cycle_indices is not None:
        cycle_rate = np.mean(
            signal_rate(
                cycle_indices, sampling_rate=sampling_rate, desired_length=desired_length
            )
        )
    
    # Modulator
    # Note: this is based on quick internal testing but could be improved
    window_size = 60 / cycle_rate  # Cycles per second

    # Window
    epochs_start = ratio_pre * window_size
    epochs_end = (1 - ratio_pre) * window_size

    return -epochs_start, epochs_end, cycle_rate