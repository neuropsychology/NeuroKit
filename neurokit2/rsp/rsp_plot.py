# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .rsp_clean import rsp_clean
from .rsp_findpeaks import rsp_findpeaks
from .rsp_rate import rsp_rate


def rsp_plot(rsp_signal, show=True):
    """RSP (respiration) signal processing.

    Examples
    ---------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=50, num=10000))
    >>> data, info = nk.rsp_process(signal, sampling_rate=1000)
    """
#    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1, sharex=True)
#    ax0.set_title("Signal and Breathing Extrema")
#    ax0.plot(self.signal)
#    ax0.scatter(self.peaks, self.signal[self.peaks])
#    ax0.scatter(self.troughs, self.signal[self.troughs])
#    ax1.set_title("Breathing Period (based on Inhalation Peaks")
#    ax1.plot(self.period)
#    ax2.set_title("Breathing Rate (based on Inhalation Peaks")
#    ax2.plot(self.rate)
#    ax3.set_title("Breathing Amplitude")
#    ax3.plot(self.amplitude)
#    plt.show()


