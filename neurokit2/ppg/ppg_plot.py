# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ppg_plot(ppg_signals, sampling_rate=None):
    """Visualize photoplethysmogram (PPG) data.

    Parameters
    ----------
    ppg_signals : DataFrame
        DataFrame obtained from `ppg_process()`.
    sampling_rate : int
        The sampling frequency of the PPG (in Hz, i.e., samples/second). Needs to be supplied if the data
        should be plotted over time in seconds. Otherwise the data is plotted over samples. Defaults to None.

    Returns
    -------
    fig
        Figure representing a plot of the processed PPG signals.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Simulate data
    >>> ppg = nk.ppg_simulate(duration=10, sampling_rate=1000, heart_rate=70)
    >>>
    >>> # Process signal
    >>> signals, info = nk.ppg_process(ppg, sampling_rate=1000)
    >>>
    >>> # Plot
    >>> nk.ppg_plot(signals) #doctest: +ELLIPSIS
    <Figure ...>

    See Also
    --------
    ppg_process

    """

    # Sanity-check input.
    if not isinstance(ppg_signals, pd.DataFrame):
        raise ValueError(
            "NeuroKit error: The `ppg_signals` argument must"
            " be the DataFrame returned by `ppg_process()`."
        )

    # X-axis
    if sampling_rate is not None:
        x_axis = np.linspace(0, ppg_signals.shape[0] / sampling_rate, ppg_signals.shape[0])
    else:
        x_axis = np.arange(0, ppg_signals.shape[0])

    # Prepare figure
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    if sampling_rate is not None:
        ax0.set_xlabel("Time (seconds)")
        ax1.set_xlabel("Time (seconds)")
    elif sampling_rate is None:
        ax0.set_xlabel("Samples")
        ax1.set_xlabel("Samples")

    fig.suptitle("Photoplethysmogram (PPG)", fontweight="bold")
    plt.subplots_adjust(hspace=0.4)

    # Plot cleaned and raw PPG
    ax0.set_title("Raw and Cleaned Signal")
    ax0.plot(x_axis, ppg_signals["PPG_Raw"], color="#B0BEC5", label="Raw", zorder=1)
    ax0.plot(x_axis, ppg_signals["PPG_Clean"], color="#FB1CF0", label="Cleaned", zorder=1, linewidth=1.5)

    # Plot peaks
    peaks = np.where(ppg_signals["PPG_Peaks"] == 1)[0]
    ax0.scatter(x_axis[peaks], ppg_signals["PPG_Clean"][peaks], color="#D60574", label="Peaks", zorder=2)
    ax0.legend(loc="upper right")

    # Rate
    ax1.set_title("Heart Rate")
    ppg_rate_mean = ppg_signals["PPG_Rate"].mean()
    ax1.plot(x_axis, ppg_signals["PPG_Rate"], color="#FB661C", label="Rate", linewidth=1.5)
    ax1.axhline(y=ppg_rate_mean, label="Mean", linestyle="--", color="#FBB41C")
    ax1.legend(loc="upper right")

    return fig
