# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def emg_plot(emg_signals, sampling_rate=None):
    """Visualize electromyography (EMG) data.

    Parameters
    ----------
    emg_signals : DataFrame
        DataFrame obtained from `emg_process()`.
    sampling_rate : int
        The sampling frequency of the EMG (in Hz, i.e., samples/second). Needs to be supplied if the
        data should be plotted over time in seconds. Otherwise the data is plotted over samples.
        Defaults to None.

    Returns
    -------
    fig
        Figure representing a plot of the processed emg signals.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> emg = nk.emg_simulate(duration=10, sampling_rate=1000, burst_number=3)
    >>> emg_signals, _ = nk.emg_process(emg, sampling_rate=1000)
    >>> fig = nk.emg_plot(emg_signals)
    >>> fig #doctest: +SKIP

    See Also
    --------
    ecg_process

    """
    # Mark onsets, offsets, activity
    onsets = np.where(emg_signals["EMG_Onsets"] == 1)[0]
    offsets = np.where(emg_signals["EMG_Offsets"] == 1)[0]

    # Sanity-check input.
    if not isinstance(emg_signals, pd.DataFrame):
        raise ValueError(
            "NeuroKit error: The `emg_signals` argument must"
            " be the DataFrame returned by `emg_process()`."
        )

    # Determine what to display on the x-axis, mark activity.
    if sampling_rate is not None:
        x_axis = np.linspace(0, emg_signals.shape[0] / sampling_rate, emg_signals.shape[0])
    else:
        x_axis = np.arange(0, emg_signals.shape[0])

    # Prepare figure.
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    if sampling_rate is not None:
        ax1.set_xlabel("Time (seconds)")
    elif sampling_rate is None:
        ax1.set_xlabel("Samples")

    fig.suptitle("Electromyography (EMG)", fontweight="bold")
    plt.subplots_adjust(hspace=0.2)

    # Plot cleaned and raw EMG.
    ax0.set_title("Raw and Cleaned Signal")
    ax0.plot(x_axis, emg_signals["EMG_Raw"], color="#B0BEC5", label="Raw", zorder=1)
    ax0.plot(x_axis, emg_signals["EMG_Clean"], color="#FFC107", label="Cleaned", zorder=1, linewidth=1.5)
    ax0.legend(loc="upper right")

    # Plot Amplitude.
    ax1.set_title("Muscle Activation")
    ax1.plot(x_axis, emg_signals["EMG_Amplitude"], color="#FF9800", label="Amplitude", linewidth=1.5)

    # Shade activity regions.
    activity_signal = _emg_plot_activity(emg_signals, onsets, offsets)
    ax1.fill_between(
        x_axis,
        emg_signals["EMG_Amplitude"],
        activity_signal,
        where=emg_signals["EMG_Amplitude"] > activity_signal,
        color="#f7c568",
        alpha=0.5,
        label=None,
    )

    # Mark onsets and offsets.
    ax1.scatter(x_axis[onsets], emg_signals["EMG_Amplitude"][onsets], color="#f03e65", label=None, zorder=3)
    ax1.scatter(x_axis[offsets], emg_signals["EMG_Amplitude"][offsets], color="#f03e65", label=None, zorder=3)

    if sampling_rate is not None:
        onsets = onsets / sampling_rate
        offsets = offsets / sampling_rate

    for i, j in zip(list(onsets), list(offsets)):
        ax1.axvline(i, color="#4a4a4a", linestyle="--", label=None, zorder=2)
        ax1.axvline(j, color="#4a4a4a", linestyle="--", label=None, zorder=2)
    ax1.legend(loc="upper right")

    plt.show()
    return fig


# =============================================================================
# Internals
# =============================================================================
def _emg_plot_activity(emg_signals, onsets, offsets):

    activity_signal = pd.Series(np.full(len(emg_signals), np.nan))
    activity_signal[onsets] = emg_signals["EMG_Amplitude"][onsets].values
    activity_signal[offsets] = emg_signals["EMG_Amplitude"][offsets].values
    activity_signal = activity_signal.fillna(method="backfill")

    if np.any(activity_signal.isna()):
        index = np.min(np.where(activity_signal.isna())) - 1
    value_to_fill = activity_signal[index]
    activity_signal = activity_signal.fillna(value_to_fill)

    return activity_signal
