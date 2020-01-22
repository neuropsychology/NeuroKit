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
    >>> signals = nk.emg_process(emg, sampling_rate=1000)
    >>> nk.emg_plot(signals)

    See Also
    --------
    ecg_process
    """
    # Sanity-check input.
    if not isinstance(emg_signals, pd.DataFrame):
        print("NeuroKit error: The `emg_signals` argument must be the "
              "DataFrame returned by `emg_process()`.")
        return

    # Determine what to display on the x-axis.
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
             label="Amplitude", linewidth=1.5)
    ax1.legend(loc="upper right")

    plt.show()
    return fig
