# - * - coding: utf-8 - * -
import matplotlib.pyplot as plt
import numpy as np

from ..epochs import epochs_create, epochs_to_df
from ..signal import signal_rate
from .ecg_peaks import ecg_peaks


def ecg_segment(ecg_cleaned, rpeaks=None, sampling_rate=1000, show=False):
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
        If ``True``, will return a plot of heartbeats. Defaults to ``False``.

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
        fig = _ecg_segment_plot(heartbeats, ytitle="ECG", heartrate=average_hr)
    if show == "return":
        return fig

    return heartbeats


# =============================================================================
# Internals
# =============================================================================
def _ecg_segment_plot(heartbeats, ytitle="ECG", heartrate=0):
    df = epochs_to_df(heartbeats)
    # Average heartbeat
    mean_heartbeat = df.drop(["Index", "Label"], axis=1).groupby("Time").mean()
    df_pivoted = df.pivot(index="Time", columns="Label", values="Signal")

    # Prepare plot
    fig = plt.figure()

    plt.title(f"Individual Heart Beats (average heart rate: {heartrate:0.1f} bpm)")
    plt.xlabel("Time (s)")
    plt.ylabel(ytitle)

    # Add Vertical line at 0
    plt.axvline(x=0, color="grey", linestyle="--")

    # Plot average heartbeat
    plt.plot(mean_heartbeat.index, mean_heartbeat, color="red", linewidth=10)

    # Plot all heartbeats
    plt.plot(df_pivoted, color="grey", linewidth=2 / 3)

    return fig


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
    m = heart_rate / 60

    # Window
    epochs_start = -0.35 / m
    epochs_end = 0.5 / m

    # Adjust for high heart rates
    if heart_rate >= 80:
        c = 0.1
        epochs_start = epochs_start - c
        epochs_end = epochs_end + c

    return epochs_start, epochs_end, heart_rate
