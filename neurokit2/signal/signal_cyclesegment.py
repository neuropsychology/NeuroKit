# - * - coding: utf-8 - * -
import matplotlib.pyplot as plt
import numpy as np

from ..epochs import epochs_create, epochs_to_df
from ..signal.signal_rate import signal_rate

def signal_cyclesegment(signal_cleaned, cycle_indices, ratio_pre=0.5, sampling_rate=1000, show=False, signal_name="signal", **kwargs):
    """**Segment a signal into individual cycles**

    Segment a signal (e.g. ECG, PPG, respiratory) into individual cycles (e.g. heartbeats, pulse waves, breaths).

    Parameters
    ----------
    signal_cleaned : Union[list, np.array, pd.Series]
        The cleaned signal channel, such as that returned by ``ppg_clean()`` or ``ecg_clean()``.
    cycle_indices : dict
        The samples indicating individual cycles (such as PPG peaks or ECG R-peaks), such as a dict
        returned by ``ppg_peaks()``.
    ratio_pre : float
        The proportion of the cycle window which takes place before the cycle index (e.g. the proportion of the
        interbeat interval which takes place before the PPG pulse peak or ECG R peak).
    sampling_rate : int
        The sampling frequency of ``signal_cleaned`` (in Hz, i.e., samples/second). Defaults to 1000.
    show : bool
        If ``True``, will return a plot of cycles. Defaults to ``False``. If "return", returns
        the axis of the plot.
    signal_name : str
        The name of the signal (only used for plotting).
    **kwargs
        Other arguments to be passed.

    Returns
    -------
    dict
        A dict containing DataFrames for all segmented cycles.
    average_cycle_rate : float
        The average cycle rate (e.g. heart rate, respiratory rate) (in Hz, i.e., samples/second)

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
      signals, peaks = nk.ppg_peaks(ppg_cleaned, sampling_rate=sampling_rate)
      peaks = peaks["PPG_Peaks"]
      heartbeats = nk.signal_cyclesegment(ppg_cleaned, peaks, sampling_rate=sampling_rate)

    """

    if len(signal_cleaned) < sampling_rate * 4:
        raise ValueError("The data length is too small to be segmented.")

    epochs_start, epochs_end, average_cycle_rate = _segment_window(
        cycle_indices=cycle_indices,
        sampling_rate=sampling_rate,
        desired_length=len(signal_cleaned),
        ratio_pre,
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
    after_last_index = cycles[last_cycle_key]["Index"] < len(signal_cleaned)
    cycles[last_cycle_key].loc[after_last_index, "Signal"] = np.nan

    # Plot or return plot axis
    if show is not False:
        ax = _segment_plot(cycles, cyclerate=average_cycle_rate, signal_name=signal_name, **kwargs)
    if show == "return":
        return ax

    return cycles, average_cycle_rate


# =============================================================================
# Internals
# =============================================================================
def _segment_plot(cycles, cyclerate=0, signal_name="signal", color="#F44336", ax=None):
    df = epochs_to_df(cycles)

    # Get main signal column name
    col = "Signal"

    # Average cycle shape
    mean_cycle = df.groupby("Time")[[col]].mean()
    df_pivoted = df.pivot(index="Time", columns="Label", values=col)

    # Prepare plot
    if ax is None:
        _, ax = plt.subplots()
    signal_name = signal_name.lower()
    if signal_name in ["ecg","ppg"]:
        cycle_name = "beat"
        rate_name = "heart rate"
        rate_unit = "bpm"
    elif signal_name == "rsp":
        cycle_name = "breath"
        rate_name = "respiratory rate"
        rate_unit = "breaths per min"
    elif signal_name == "signal":
        cycle_name = "cycle"
        rate_name = "cycle rate"
        rate_unit = "per min"

    ax.set_title(f"Individual {cycle_name}s (average {rate_name}: {cyclerate:0.1f} {rate_unit})")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(signal_name)

    # Add Vertical line at 0
    ax.axvline(x=0, color="grey", linestyle="--")

    # Plot average cycle
    ax.plot(
        mean_cycle.index,
        mean_cycle,
        color=color,
        linewidth=7,
        label=f"Average {cycle_name} shape",
        zorder=1,
    )

    # Alpha of individual cycles decreases with more cycles
    n_cycles = df_pivoted.shape[1]
    if n_cycles <= 1:
        alpha = 1.0
    else:
        alpha = 1 / np.log2(np.log2(1 + n_cycles))

    # Plot all cycles
    ax.plot(df_pivoted, color="grey", linewidth=alpha, alpha=alpha, zorder=2)

    # Legend
    ax.legend(loc="upper right")
    return ax

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