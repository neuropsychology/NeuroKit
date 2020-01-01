# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def rsp_plot(rsp_signals, sampling_rate=None):
    """Visualize respiration (RSP) data.

    Parameters
    ----------
    rsp_signals : DataFrame
        DataFrame obtained from `rsp_process`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(duration=90)
    >>> rsp_signals, info = nk.rsp_process(rsp, sampling_rate=1000)
    >>> nk.rsp_plot(rsp_signals, sampling_rate=1000)

    See Also
    --------
    rsp_process
    """
    # Get x-axis
    if sampling_rate is not None:
        x_axis = np.linspace(0, len(rsp_signals) / sampling_rate, len(rsp_signals))
    else:
        x_axis = np.arange(0, len(rsp_signals))

    # Extract peaks and troughs
    peaks = np.where(rsp_signals["RSP_Peaks"] == 1)[0]
    troughs = np.where(rsp_signals["RSP_Troughs"] == 1)[0]

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)
    plt.subplots_adjust(hspace=0.2)
    ax0.set_title("Raw and Cleaned RPS")
    ax0.plot(x_axis, rsp_signals["RSP_Raw"], color='#607D8B', label='Raw', zorder=1)
    ax0.plot(x_axis, rsp_signals["RSP_Clean"], color='#2196F3', label='Cleaned', zorder=1)
    ax0.set_ylabel('Amplitude (m)')
    ax0.legend(loc='upper right')
    ax0.scatter(x_axis[peaks], rsp_signals["RSP_Clean"][peaks], color='red', zorder=2)
    ax0.scatter(x_axis[troughs], rsp_signals["RSP_Clean"][troughs], color='orange', zorder=2)
    ax1.set_title("Breathing Rate")
    ax1.plot(x_axis, rsp_signals["RSP_Rate"], color='#4CAF50', label='Data')
    rate_mean = [np.mean(rsp_signals["RSP_Rate"])]*len(rsp_signals["RSP_Rate"])
    ax1.plot(x_axis, rate_mean, label='Mean', linestyle='--', color='#4CAF50')
    ax1.set_ylabel('Inhales per minute')
    ax1.legend(loc='upper right')
    ax2.set_title("Breathing Amplitude")
    if "RSP_Amplitude" in list(rsp_signals.columns):
        ax2.plot(x_axis, rsp_signals["RSP_Amplitude"], color='#009688', label='Data')
        amplitude_mean = [np.mean(rsp_signals["RSP_Amplitude"])]*len(rsp_signals["RSP_Amplitude"])
        ax2.plot(x_axis, amplitude_mean, label='Mean', linestyle='--', color='#009688')
        ax2.set_ylabel('a.u.')
    if sampling_rate is not None:
        ax2.set_xlabel('Time (s)')
    else:
        ax2.set_xlabel('Data points')
    ax2.legend(loc='upper right')
    fig.suptitle('Respiration (RSP) Processing', fontweight='bold')
    plt.show()
    return fig
