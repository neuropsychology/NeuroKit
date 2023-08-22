# - * - coding: utf-8 - * -
import matplotlib.pyplot as plt
import numpy as np

from ..epochs import epochs_create, epochs_to_df
from ..signal import signal_rate
from .ecg_peaks import ecg_peaks


def ecg_segment(ecg_cleaned, rpeaks=None, sampling_rate=1000, show=False, **kwargs):
    """**Segment an ECG signal into single heartbeats**

    Segment an ECG signal into single heartbeats. Convenient for visualizing all the heart beats.

    Parameters
    ----------
    ecg_cleaned : Union[list, np.array, pd.Series]
        The cleaned ECG channel as returned by ``ecg_clean()``.
    rpeaks : dict
        The samples at which the R-peaks occur. Dict returned by ``ecg_peaks()``. Defaults to ``None``.
    sampling_rate : int
        The sampling frequency of ``ecg_cleaned`` (in Hz, i.e., samples/second). Defaults to 1000.
    show : bool
        If ``True``, will return a plot of heartbeats. Defaults to ``False``. If "return", returns
        the axis of the plot.
    **kwargs
        Other arguments to be passed.

    Returns
    -------
    dict
        A dict containing DataFrames for all segmented heartbeats.

    See Also
    --------
    ecg_clean, ecg_plot

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      ecg = nk.ecg_simulate(duration=15, sampling_rate=1000, heart_rate=80, noise = 0.05)
      @savefig p_ecg_segment.png scale=100%
      qrs_epochs = nk.ecg_segment(ecg, rpeaks=None, sampling_rate=1000, show=True)
      @suppress
      plt.close()

    """
    # Sanitize inputs
    if rpeaks is None:
        _, rpeaks = ecg_peaks(
            ecg_cleaned, sampling_rate=sampling_rate, correct_artifacts=True
        )
        rpeaks = rpeaks["ECG_R_Peaks"]

    epochs_start, epochs_end, average_hr = _ecg_segment_window(
        rpeaks=rpeaks, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned)
    )
    heartbeats = epochs_create(
        ecg_cleaned,
        rpeaks,
        sampling_rate=sampling_rate,
        epochs_start=epochs_start,
        epochs_end=epochs_end,
    )

    # pad last heartbeat with nan so that segments are equal length
    last_heartbeat_key = str(np.max(np.array(list(heartbeats.keys()), dtype=int)))
    after_last_index = heartbeats[last_heartbeat_key]["Index"] < len(ecg_cleaned)
    heartbeats[last_heartbeat_key].loc[after_last_index, "Signal"] = np.nan

    if show is not False:
        ax = _ecg_segment_plot(heartbeats, heartrate=average_hr, ytitle="ECG", **kwargs)
    if show == "return":
        return ax

    return heartbeats


# =============================================================================
# Internals
# =============================================================================
def _ecg_segment_plot(heartbeats, heartrate=0, ytitle="ECG", color="#F44336", ax=None):
    df = epochs_to_df(heartbeats)
    # Average heartbeat
    mean_heartbeat = df.drop(["Index", "Label"], axis=1).groupby("Time").mean()
    df_pivoted = df.pivot(index="Time", columns="Label", values="Signal")

    # Prepare plot
    if ax is None:
        fig, ax = plt.subplots()

    ax.set_title(f"Individual Heart Beats (average heart rate: {heartrate:0.1f} bpm)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(ytitle)

    # Add Vertical line at 0
    ax.axvline(x=0, color="grey", linestyle="--")

    # Plot average heartbeat
    ax.plot(
        mean_heartbeat.index, mean_heartbeat, color=color, linewidth=10, label="Average"
    )

    # Plot all heartbeats
    alpha = 1 / np.log2(1 + df_pivoted.shape[1])  # alpha decreases with more heartbeats
    ax.plot(df_pivoted, color="grey", linewidth=alpha)
    ax.legend(loc="upper right")

    return ax


def _ecg_segment_window(
    heart_rate=None, rpeaks=None, sampling_rate=1000, desired_length=None
):
    # Extract heart rate
    if heart_rate is not None:
        heart_rate = np.mean(heart_rate)
    if rpeaks is not None:
        heart_rate = np.mean(
            signal_rate(
                rpeaks, sampling_rate=sampling_rate, desired_length=desired_length
            )
        )

    # Modulator
    # Note: this is based on quick internal testing but could be improved
    window_size = 60 / heart_rate  # Beats per second

    # Window
    epochs_start = 1 / 3 * window_size
    epochs_end = 2 / 3 * window_size

    return -epochs_start, epochs_end, heart_rate
