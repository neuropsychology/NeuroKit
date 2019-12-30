# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def rsp_plot(rsp_signal, duration=60):
    """Visualize respiration (RSP) data.

    Parameters
    ----------
    rsp_signal : DataFrame
        DataFrame obtained from `rsp_process`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> rsp = np.cos(np.linspace(start=0, stop=50, num=10000))
    >>> rsp_signal, info = rsp_process(rsp, sampling_rate=1000)
    >>> nk.rsp_plot(rsp_signal)

    See Also
    --------
    rsp_process
    """
    # Append time stamp
    t = np.linspace(0, duration, len(rsp_signal))
    t = pd.DataFrame(t)
    rsp_signal = pd.concat([rsp_signal, t], axis=1)
    rsp_signal.columns = [*rsp_signal.columns[:-1], 'Time']

    # Extract peaks and troughs
    peaks = np.where(rsp_signal["RSP_Peaks"] == 1)[0]
    troughs = np.where(rsp_signal["RSP_Troughs"] == 1)[0]

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)
    plt.subplots_adjust(hspace=0.2)

    # Plot clean versus raw signal
    ax0.set_title("Signal and Breathing Extrema")
    ax0.plot(rsp_signal["Time"], rsp_signal["RSP_Raw"], color='grey', label='Raw', zorder=1)
    ax0.plot(rsp_signal["Time"], rsp_signal["RSP_Clean"], color='blue', label='Cleaned', zorder=1)
    ax0.set_ylabel('Amplitude (m)')
    ax0.legend(loc='upper right')
    ax0.scatter(rsp_signal["Time"][peaks], rsp_signal["RSP_Clean"][peaks], color='red', zorder=2)
    ax0.scatter(rsp_signal["Time"][troughs], rsp_signal["RSP_Clean"][troughs], color='orange', zorder=2)

    # Plot breathing rate
    ax1.set_title("Breathing Rate")
    ax1.plot(rsp_signal["Time"], rsp_signal["RSP_Rate"], color='purple', label='Data')
    rate_mean = [np.mean(rsp_signal["RSP_Rate"])]*len(rsp_signal["RSP_Rate"])
    ax1.plot(rsp_signal["Time"], rate_mean, label='Mean', linestyle='--', color='purple')
    ax1.set_ylabel('Breaths per minute (Bpm)')
    ax1.legend(loc='upper right')

    # Plot breathing amplitude
    ax2.set_title("Breathing Amplitude")
    ax2.plot(rsp_signal["Time"], rsp_signal["RSP_Amplitude"], color='brown', label='Data')
    amplitude_mean = [np.mean(rsp_signal["RSP_Amplitude"])]*len(rsp_signal["RSP_Amplitude"])
    ax2.plot(rsp_signal["Time"], amplitude_mean, label='Mean', linestyle='--', color='brown')
    ax2.set_xlabel('Time (s)')
    ax2.legend(loc='upper right')
    plt.show()
    return fig



