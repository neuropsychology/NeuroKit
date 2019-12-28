# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt


def rsp_plot(rsp_signals):
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
    >>> rsp = np.cos(np.linspace(start=0, stop=50, num=10000))
    >>> signals, info = nk.rsp_process(rsp, sampling_rate=1000)
    >>> nk.rsp_plot(signals)

    See Also
    --------
    rsp_process
    """
    peaks = np.where(rsp_signals["RSP_Peaks"] == 1)[0]
    troughs = np.where(rsp_signals["RSP_Troughs"] == 1)[0]

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax0.set_title("Signal and Breathing Extrema")
    ax0.plot(rsp_signals["RSP_Clean"])
    ax0.scatter(peaks, rsp_signals["RSP_Clean"][peaks])
    ax0.scatter(troughs, rsp_signals["RSP_Clean"][troughs])
    ax1.set_title("Breathing Rate")
    ax1.plot(rsp_signals["RSP_Rate"])
    ax2.set_title("Breathing Amplitude")
    ax2.plot(rsp_signals["RSP_Amplitude"])
    plt.show()
    return fig
