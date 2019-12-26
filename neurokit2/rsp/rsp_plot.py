# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt


def rsp_plot(rsp_summary):
    """Visualize respiration (RSP) data.

    Parameters
    ----------
    rsp_summary : DataFrame
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
    peaks = np.where(rsp_summary["RSP_Peaks"] == 1)[0]
    troughs = np.where(rsp_summary["RSP_Troughs"] == 1)[0]

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1, sharex=True)
    ax0.set_title("Signal and Breathing Extrema")
    ax0.plot(rsp_summary["RSP_Filtered"])
    ax0.scatter(peaks, rsp_summary["RSP_Filtered"][peaks])
    ax0.scatter(troughs, rsp_summary["RSP_Filtered"][troughs])
    ax2.set_title("Breathing Rate")
    ax2.plot(rsp_summary["RSP_Rate"])
    ax3.set_title("Breathing Amplitude")
    ax3.plot(rsp_summary["RSP_Amplitude"])
    plt.show()
