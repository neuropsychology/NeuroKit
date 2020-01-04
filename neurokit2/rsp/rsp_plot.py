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
    >>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(duration=90, respiratory_rate=15)
    >>> signals, info = nk.rsp_process(rsp, sampling_rate=1000)
    >>> nk.rsp_plot(signals)

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


    # Initiate
    if "RSP_Amplitude" in list(rsp_signals.columns):
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)
        if sampling_rate is not None:
            ax2.set_xlabel('Time (s)')
    else:
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        if sampling_rate is not None:
            ax1.set_xlabel('Time (s)')

    plt.subplots_adjust(hspace=0.2)

    # Upper
    ax0.set_title("Raw and Cleaned RSP")
    fig.suptitle('Respiration (RSP) Processing', fontweight='bold')
#    ax0.set_ylabel('Amplitude (m)')

    # Plot cleaned and raw
    ax0.plot(x_axis, rsp_signals["RSP_Raw"], color='#B0BEC5', label='Raw', zorder=1)
    ax0.plot(x_axis, rsp_signals["RSP_Clean"], color='#2196F3', label='Cleaned', zorder=1)
    ax0.legend(loc='upper right')

    # Peaks and troughs
    ax0.scatter(x_axis[peaks], rsp_signals["RSP_Clean"][peaks], color='red', zorder=2)
    ax0.scatter(x_axis[troughs], rsp_signals["RSP_Clean"][troughs], color='orange', zorder=2)

    # Rate
    ax1.set_title("Breathing Rate")
#    ax1.set_ylabel('Inhales per minute')
    ax1.plot(x_axis, rsp_signals["RSP_Rate"], color='#4CAF50', label='Rate')
    rate_mean = [np.mean(rsp_signals["RSP_Rate"])]*len(rsp_signals["RSP_Rate"])
    ax1.plot(x_axis, rate_mean, label='Mean', linestyle='--', color='#4CAF50')
    ax1.legend(loc='upper right')

    if "RSP_Amplitude" in list(rsp_signals.columns):
        ax2.set_title("Breathing Amplitude")
#        ax2.set_ylabel('a.u.')

        ax2.plot(x_axis, rsp_signals["RSP_Amplitude"], color='#009688', label='Amplitude')
        amplitude_mean = [np.mean(rsp_signals["RSP_Amplitude"])]*len(rsp_signals["RSP_Amplitude"])
        ax2.plot(x_axis, amplitude_mean, label='Mean', linestyle='--', color='#009688')
        ax2.legend(loc='upper right')


    plt.show()
    return fig
