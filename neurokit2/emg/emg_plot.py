# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def emg_plot(emg_signals, sampling_rate=None):
    """Visualize electromyography (EMG) data.

    Parameters
    ----------
    emg_signals : DataFrame
        DataFrame obtained from `emg_process()`.
    sampling_rate : int
        The sampling frequency of the EMG (in Hz, i.e., samples/second). Needs
        to be supplied if the data should be plotted over time in seconds.
        Otherwise the data is plotted over samples. Defaults to None.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> emg = nk.emg_simulate(duration=10, sampling_rate=1000, n_bursts=3)
    >>> emg_signals, _ = nk.emg_process(emg, sampling_rate=1000)
    >>> nk.emg_plot(emg_signals)

    See Also
    --------
    ecg_process
    """
    # Mark onsets, offsets, activity
    onsets = np.where(emg_signals["EMG_Onsets"] == 1)[0]
    offsets = np.where(emg_signals["EMG_Offsets"] == 1)[0]

    # Sanity-check input.
    if not isinstance(emg_signals, pd.DataFrame):
        print("NeuroKit error: The `emg_signals` argument must be the "
              "DataFrame returned by `emg_process()`.")

    # Determine what to display on the x-axis, mark activity.
    if sampling_rate is not None:
        x_axis = np.linspace(0, emg_signals.shape[0] / sampling_rate,
                             emg_signals.shape[0])
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
    ax0.plot(x_axis, emg_signals["EMG_Raw"], color='#B0BEC5', label='Raw',
             zorder=1)
    ax0.plot(x_axis, emg_signals["EMG_Clean"], color='#FFC107',
             label="Cleaned", zorder=1, linewidth=1.5)
    ax0.legend(loc="upper right")

    # Plot Amplitude.
    ax1.set_title("Muscle Activation")
    ax1.plot(x_axis, emg_signals["EMG_Amplitude"], color="#FF9800",
             label=None, linewidth=1.5)

    # Shade activity regions.
    activity_signal = _emg_plot_activity(emg_signals, onsets, offsets)
    ax1.fill_between(x_axis, emg_signals["EMG_Amplitude"], activity_signal,
                     where=emg_signals["EMG_Amplitude"] > activity_signal,
                     color='#f7c568', alpha=0.5, label='Regions of Activity')

    # Mark onsets and offsets.
    if sampling_rate is not None:
        onsets = onsets / sampling_rate
        offsets = offsets / sampling_rate
    else:
        onsets = onsets
        offsets = offsets

    for i, j in zip(list(onsets), list(offsets)):
        ax1.axvline(i, color='#9da1a6', linestyle='--', label='Onsets', zorder=3)
        ax1.axvline(j, color='#c1c5c9', linestyle='--', label='Offsets', zorder=3)

    # Remove duplicate labels
    handles, labels = ax1.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)

    ax1.legend(*[*zip(*{l:h for h, l in zip(*ax1.get_legend_handles_labels())}.items())][::-1], loc='upper right')

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
