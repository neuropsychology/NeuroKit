# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def rsp_plot(rsp_data):
    """Visualize respiration data.

    Parameters
    ----------
    rsp_data : DataFrame
        DataFrame containing respiration data, as obtained from `rsp_process()`.

    Examples
    ---------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=50, num=10000))
    >>> rsp_data, info = nk.rsp_process(signal, sampling_rate=1000)
    >>> nk.rsp_plot(rsp_data)

    See Also
    --------
    rsp_process
    """
    peaks = np.where(rsp_data["RSP_Peaks"] == 1)[0]
    troughs = np.where(rsp_data["RSP_Troughs"] == 1)[0]

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1, sharex=True)
    ax0.set_title("Signal and Breathing Extrema")
    ax0.plot(rsp_data["RSP_Filtered"])
    ax0.scatter(peaks, rsp_data["RSP_Filtered"][peaks])
    ax0.scatter(troughs, rsp_data["RSP_Filtered"][troughs])
    ax1.set_title("Breathing Period")
    ax1.plot(rsp_data["RSP_Period"])
    ax2.set_title("Breathing Rate")
    ax2.plot(rsp_data["RSP_Rate"])
    ax3.set_title("Breathing Amplitude")
    ax3.plot(rsp_data["RSP_Amplitude"])
    plt.show()


