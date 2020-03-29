# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec

from ..ecg import ecg_findpeaks
from .ecg_segment import ecg_segment
from ..epochs import epochs_to_df
from ..epochs import epochs_create

def ecg_plot(ecg_signals, sampling_rate=None):
    """Visualize ECG data.

    Parameters
    ----------
    ecg_signals : DataFrame
        DataFrame obtained from `ecg_process()`.
    sampling_rate : int
        The sampling frequency of the ECG (in Hz, i.e., samples/second). Needs
        to be supplied if the data should be plotted over time in seconds.
        Otherwise the data is plotted over samples. Defaults to None.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> ecg = nk.ecg_simulate(duration=15, sampling_rate=1000, heart_rate=80)
    >>> signals, info = nk.ecg_process(ecg, sampling_rate=1000)
    >>> nk.ecg_plot(signals, sampling_rate=1000)

    See Also
    --------
    ecg_process
    """
    # Sanity-check input.
    if not isinstance(ecg_signals, pd.DataFrame):
        print("NeuroKit error: The `ecg_signals` argument must be the "
              "DataFrame returned by `ecg_process()`.")

    # Extract R-peaks.
    peaks = np.where(ecg_signals["ECG_R_Peaks"] == 1)[0]

    # Prepare figure and set axes.
    if sampling_rate is not None:
        x_axis = np.linspace(0, ecg_signals.shape[0] / sampling_rate,
                             ecg_signals.shape[0])
        gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[1-1/np.pi, 1/np.pi])
        fig = plt.figure(constrained_layout=False)
        ax0 = fig.add_subplot(gs[0, :-1])
        ax1 = fig.add_subplot(gs[1, :-1])
        ax2 = fig.add_subplot(gs[:, -1])
        ax0.set_xlabel("Time (seconds)")
        ax1.set_xlabel("Time (seconds)")
        ax2.set_xlabel("Time (seconds)")
    elif sampling_rate is None:
        x_axis = np.arange(0, ecg_signals.shape[0])
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax0.set_xlabel("Samples")
        ax1.set_xlabel("Samples")

    fig.suptitle("Electrocardiogram (ECG)", fontweight="bold")
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    # Plot cleaned and raw ECG as well as R-peaks.
    ax0.set_title("Raw and Cleaned Signal")

    ax0.plot(x_axis, ecg_signals["ECG_Raw"], color='#B0BEC5', label='Raw',
             zorder=1)
    ax0.plot(x_axis, ecg_signals["ECG_Clean"], color='#E91E63',
             label="Cleaned", zorder=1, linewidth=1.5)
    ax0.scatter(x_axis[peaks], ecg_signals["ECG_Clean"][peaks],
                color="#FFC107", label="R-peaks", zorder=2)

    # Plot heart rate.
    ax1.set_title("Heart Rate")
    ax1.set_ylabel("Beats per minute (bpm)")

    ax1.plot(x_axis, ecg_signals["ECG_Rate"],
             color="#FF5722", label="Rate", linewidth=1.5)
    rate_mean = ecg_signals["ECG_Rate"].mean()
    ax1.axhline(y=rate_mean, label="Mean", linestyle="--", color="#FF9800")

    ax1.legend(loc="upper right")

    # Plot individual heart beats.
    if sampling_rate is not None:
        ax2.set_title("Individual Heart Beats")

        heartbeats = ecg_segment(ecg_signals["ECG_Clean"], peaks, sampling_rate)
        heartbeats = epochs_to_df(heartbeats)

        heartbeats_pivoted = heartbeats.pivot(index='Time',
                                              columns='Label',
                                              values='Signal')

        ax2.plot(heartbeats_pivoted)

        cmap = iter(plt.cm.YlOrRd(
                np.linspace(0, 1, num=int(heartbeats["Label"].nunique()))))  # Aesthetics of heart beats

        lines = []
        for x, color in zip(heartbeats_pivoted, cmap):
            line, = ax2.plot(heartbeats_pivoted[x], color=color)
            lines.append(line)

    return fig
