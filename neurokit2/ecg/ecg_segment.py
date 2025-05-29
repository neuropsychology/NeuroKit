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

    if len(ecg_cleaned) < sampling_rate * 4:
        raise ValueError("The data length is too small to be segmented.")

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

    # Pad last heartbeats with nan so that segments are equal length
    last_heartbeat_key = str(np.max(np.array(list(heartbeats.keys()), dtype=int)))
    after_last_index = heartbeats[last_heartbeat_key]["Index"] >= len(ecg_cleaned)
    for col in ["Signal", "ECG_Raw", "ECG_Clean"]:
        if col in heartbeats[last_heartbeat_key].columns:
            heartbeats[last_heartbeat_key].loc[after_last_index, col] = np.nan

    # Plot or return plot axis (feature meant to be used internally in ecg_plot)
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

    # Get main signal column name
    col = [c for c in ["Signal", "ECG_Raw", "ECG_Clean"] if c in df.columns][-1]

    # Average heartbeat
    mean_heartbeat = df.groupby("Time")[[col]].mean()
    df_pivoted = df.pivot(index="Time", columns="Label", values=col)

    # Prepare plot
    if ax is None:
        _, ax = plt.subplots()

    ax.set_title(f"Individual Heart Beats (average heart rate: {heartrate:0.1f} bpm)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(ytitle)

    # Add Vertical line at 0
    ax.axvline(x=0, color="grey", linestyle="--")

    # Plot average heartbeat
    ax.plot(
        mean_heartbeat.index,
        mean_heartbeat,
        color=color,
        linewidth=7,
        label="Average beat shape",
        zorder=1,
    )

    # Alpha of individual beats decreases with more heartbeats
    alpha = 1 / np.log2(np.log2(1 + df_pivoted.shape[1]))

    # Plot all heartbeats
    ax.plot(df_pivoted, color="grey", linewidth=alpha, alpha=alpha, zorder=2)

    # Plot individual waves
    for wave in [
        ("P", "#3949AB"),
        ("Q", "#1E88E5"),
        ("S", "#039BE5"),
        ("T", "#00ACC1"),
    ]:
        wave_col = f"ECG_{wave[0]}_Peaks"
        if wave_col in df.columns:
            ax.scatter(
                df["Time"][df[wave_col] == 1],
                df[col][df[wave_col] == 1],
                color=wave[1],
                marker="+",
                label=f"{wave[0]}-waves",
                zorder=3,
            )

    # Legend
    ax.legend(loc="upper right")
    return ax


def _ecg_segment_window(
    heart_rate=None,
    rpeaks=None,
    sampling_rate=1000,
    desired_length=None,
    ratio_pre=0.35,
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
    epochs_start = ratio_pre * window_size
    epochs_end = (1 - ratio_pre) * window_size

    return -epochs_start, epochs_end, heart_rate
