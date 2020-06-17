import numpy as np


def _eog_simulate_blink(sampling_rate=1000, length=None, size=0.05, time_peak=6.6, rise=1.2, decay1=2.3, decay2=8.0):
    """Simulate a canonical blink from vertical EOG

    Examples
    --------
    >>> blink = _eog_simulate_blink(sampling_rate=100)
    >>> nk.signal_plot(blink, sampling_rate=100)

    """
    if length is None:
        length = int(sampling_rate)

    x = np.linspace(0, 20, num=length)
    gt = np.exp(-((x - time_peak) ** 2) / (2 * rise ** 2))
    ht = np.exp(-x / decay1) + np.exp(-x / decay2)
    ft = np.convolve(gt, ht)
    ft = ft[0 : len(x)]
    y = size * ft
    return y