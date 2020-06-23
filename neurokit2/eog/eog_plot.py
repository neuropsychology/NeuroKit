# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def eog_plot(eog_signals, sampling_rate=None):
    """Visualize ECG data.

    Parameters
    ----------
    eog_signals : DataFrame
        DataFrame obtained from `eog_process()`.
    sampling_rate : int
        The sampling frequency of the EOG (in Hz, i.e., samples/second). Needs to be supplied if the data
        should be plotted over time in seconds. Otherwise the data is plotted over samples. Defaults to None.
        Must be specified to plot artifacts.

    Returns
    -------
    fig
        Figure representing a plot of the processed EOG signals.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Get data
    >>> eog_signal = nk.data('eog_100hz')
    >>>
    >>> # Process
    >>> eog_signals, info = nk.eog_process(eog_signal, sampling_rate=100)
    >>>
    >>> nk.eog_plot(signals, sampling_rate=100) #doctest: +ELLIPSIS
    <Figure ...>

    See Also
    --------
    eog_process

    """

    # Sanity-check input.
    if not isinstance(eog_signals, pd.DataFrame):
        print("NeuroKit error: The `eog_signals` argument must be the DataFrame returned by `eog_process()`.")

    # X-axis
    if sampling_rate is not None:
        x_axis = np.linspace(0, eog_signals.shape[0] / sampling_rate, eog_signals.shape[0])
    else:
        x_axis = np.arange(0, eog_signals.shape[0])

    # Prepare figure
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    if sampling_rate is not None:
        ax0.set_xlabel("Time (seconds)")
        ax1.set_xlabel("Time (seconds)")
    elif sampling_rate is None:
        ax0.set_xlabel("Samples")
        ax1.set_xlabel("Samples")

    fig.suptitle("Electrooculography (EOG)", fontweight="bold")
    plt.subplots_adjust(hspace=0.2)

    # Plot cleaned and raw EOG
    ax0.set_title("Raw and Cleaned Signal")
    ax0.plot(x_axis, eog_signals["EOG_Raw"], color="#B0BEC5", label="Raw", zorder=1)
    ax0.plot(x_axis, eog_signals["EOG_Clean"], color="#49A4FD", label="Cleaned", zorder=1, linewidth=1.5)
    ax0.set_ylabel("Amplitude (mV)")

    # Plot blinks
    blinks = np.where(eog_signals["EOG_Blinks"] == 1)[0]
    ax0.scatter(x_axis[blinks], eog_signals["EOG_Clean"][blinks], color="#0146D7", label="Blinks", zorder=2)
    ax0.legend(loc="upper right")

    # Rate
    ax1.set_title("Blink Rate")
    ax1.set_ylabel("Blinks per minute")
    blink_rate_mean = eog_signals["EOG_Rate"].mean()
    ax1.plot(x_axis, eog_signals["EOG_Rate"], color="#9C5AFF", label="Rate", linewidth=1.5)
    ax1.axhline(y=blink_rate_mean, label="Mean", linestyle="--", color="#CEAFFF")
    ax1.legend(loc="upper right")

    return fig
