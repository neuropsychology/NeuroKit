# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..ecg import ecg_findpeaks
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
    >>> nk.ecg_plot(signals)

    See Also
    --------
    ecg_process
    """
    # Sanity-check input.
    if not isinstance(ecg_signals, pd.DataFrame):
        print("NeuroKit error: The `ecg_signals` argument must be the "
              "DataFrame returned by `ecg_process()`.")

    # Determine what to display on the x-axis.
    if sampling_rate is not None:
        x_axis = np.linspace(0, ecg_signals.shape[0] / sampling_rate,
                             ecg_signals.shape[0])
    else:
        x_axis = np.arange(0, ecg_signals.shape[0])

    # Extract R-peaks.
    peaks = np.where(ecg_signals["ECG_R_Peaks"] == 1)[0]

    # Prepare figure.
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=False)
    if sampling_rate is not None:
        ax0.set_xlabel("Time (seconds)")
        ax1.set_xlabel("Time (seconds)")
        ax2.set_xlabel("Time (seconds)")
    elif sampling_rate is None:
        ax0.set_xlabel("Samples")
        ax1.set_xlabel("Samples")
        ax2.set_xlabel("Samples")

    fig.suptitle("Electrocardiogram (ECG)", fontweight="bold")
    plt.subplots_adjust(hspace=0.5)

    # Plot cleaned and raw ECG as well as R-peaks.
    ax0.set_title("Raw and Cleaned Signal")

    ax0.plot(x_axis, ecg_signals["ECG_Raw"], color='#B0BEC5', label='Raw',
             zorder=1)
    ax0.plot(x_axis, ecg_signals["ECG_Clean"], color='#E91E63',
             label="Cleaned", zorder=1, linewidth=1.5)
    ax0.scatter(x_axis[peaks], ecg_signals["ECG_Clean"][peaks], color="#FFC107",
                label="R-peaks", zorder=2)

    ax0.legend(loc="upper right")

    # Plot heart rate.
    ax1.set_title("Heart Rate")
    ax1.set_ylabel("Beats per minute (bpm)")

    ax1.plot(x_axis, ecg_signals["ECG_Rate"], color="#FF5722", label="Rate", linewidth=1.5)
    rate_mean = ecg_signals["ECG_Rate"].mean()
    ax1.axhline(y=rate_mean, label="Mean", linestyle="--", color="#FF9800")

    ax1.legend(loc="upper right")

    # Plot individual heart beats
    ax2.set_title("Individual Heart Beats")

    heartbeats = _ecg_plot_heartbeats(ecg=ecg_signals["ECG_Clean"], peaks=peaks, ax=ax2)
    ax2.plot(heartbeats["Time"].values, heartbeats["Signal"].values, legend="")

    ax2.legend(loc="upper right")
    ax2.legend(fontsize='x-small')

    plt.show()
    return fig




# =============================================================================
# Internals
# =============================================================================
def _ecg_plot_heartbeats(ecg, peaks, ax):
    # Extract heart beats
    heartbeats = epochs_create(ecg, events=peaks, epochs_duration=0.6, epochs_start=-0.3)
    heartbeats = epochs_to_df(heartbeats)

    return heartbeats
