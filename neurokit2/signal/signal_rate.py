# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def signal_rate(peaks):
    """
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=10, num=1000)) # Low freq
    >>> signal += np.cos(np.linspace(start=0, stop=100, num=1000)) # High freq
    """
    return(peaks)

#    period = np.ediff1d(self.peaks, to_begin=0) / self.sfreq
#    period[0] = np.mean(period)
#    rate = 60 / period
#    # TODO: normalize amplitude?
#    amplitude = self.peaks - self.troughs
#
#    # Interpolate all statistics to length of the breathing signal.
#    nsamps = len(self.signal)
#    self.period = signal_interpolate(self.peaks, x_axis=period, length=nsamps)
#    self.rate = signal_interpolate(self.peaks, x_axis=rate, length=nsamps)
#    self.amplitude = signal_interpolate(self.peaks, x_axis=amplitude, length=nsamps)
#    return(peaks)