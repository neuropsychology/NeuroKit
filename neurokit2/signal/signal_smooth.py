# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal

from ..stats import fit_loess


def signal_smooth(signal, method="convolution", kernel="boxzen", size=10, alpha=0.1):
    """
    Signal smoothing.

    Signal smoothing can be achieved using either the convolution of a filter kernel with the input
    signal to compute the smoothed signal (Smith, 1997) or a LOESS regression.

    Parameters
    ----------
    signal : list, array or Series
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        Can be one of 'convolution' (default) or 'loess'.
    kernel : str, array
        Only used if `method` is 'convolution'. Type of kernel to use; if array, use directly as the kernel. Can be one
        of 'median', 'boxzen', 'boxcar', 'triang', 'blackman', 'hamming',
        'hann', 'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris',
        'nuttall', 'barthann', 'kaiser' (needs beta), 'gaussian' (needs std),
        'general_gaussian' (needs power, width), 'slepian' (needs width) or
        'chebwin' (needs attenuation).
    size : int
        Only used if `method` is 'convolution'. Size of the kernel; ignored if kernel is an array.
    alpha : float
        Only used if `method` is 'loess'. The parameter which controls the degree of smoothing.

    Returns
    -------
    array
        Smoothed signal.


    See Also
    ---------
    fit_loess

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=10, num=1000))
    >>> distorted = nk.signal_distort(signal, noise_amplitude=[0.3, 0.2, 0.1, 0.05], noise_frequency=[5, 10, 50, 100])
    >>>
    >>> size = len(signal)/100
    >>> signals = pd.DataFrame({"Raw": distorted, "Median": nk.signal_smooth(distorted, kernel='median', size=size-1), "BoxZen": nk.signal_smooth(distorted, kernel='boxzen', size=size), "Triang": nk.signal_smooth(distorted, kernel='triang', size=size), "Blackman": nk.signal_smooth(distorted, kernel='blackman', size=size), "Loess_01": nk.signal_smooth(distorted, method='loess', alpha=0.1), "Loess_02": nk.signal_smooth(distorted, method='loess', alpha=0.2), "Loess_05": nk.signal_smooth(distorted, method='loess', alpha=0.5)})
    >>> fig = signals.plot()
    >>> fig_magnify = signals[50:150].plot()  # Magnify
    >>> fig_magnify #doctest: +SKIP

    References
    ----------
    - Smith, S. W. (1997). The scientist and engineer's guide to digital signal
    processing.

    """
    if isinstance(signal, pd.Series):
        signal = signal.values

    length = len(signal)

    if isinstance(kernel, str) is False:
        raise TypeError("NeuroKit error: signal_smooth(): 'kernel' should be a string.")

    # Check length.
    if size > length or size < 1:
        raise TypeError("NeuroKit error: signal_smooth(): 'size' should be between 1 and length of the signal.")

    method = method.lower()

    # LOESS
    if method in ["loess", "lowess"]:
        smoothed = fit_loess(signal, alpha=alpha)

    # Convolution
    else:
        if kernel == "boxzen":
            # hybrid method
            # 1st pass - boxcar kernel
            x = _signal_smoothing(signal, kernel="boxcar", size=size)

            # 2nd pass - parzen kernel
            smoothed = _signal_smoothing(x, kernel="parzen", size=size)

        elif kernel == "median":
            smoothed = _signal_smoothing_median(signal, size)

        else:
            smoothed = _signal_smoothing(signal, kernel=kernel, size=size)

    return smoothed


# =============================================================================
# Internals
# =============================================================================
def _signal_smoothing_median(signal, size=5):

    # Enforce odd kernel size.
    if size % 2 == 0:
        size += 1

    smoothed = scipy.signal.medfilt(signal, kernel_size=int(size))
    return smoothed


def _signal_smoothing(signal, kernel="boxcar", size=5):

    # Get window.
    size = int(size)
    window = scipy.signal.get_window(kernel, size)
    w = window / window.sum()

    # Extend signal edges to avoid boundary effects.
    x = np.concatenate((signal[0] * np.ones(size), signal, signal[-1] * np.ones(size)))

    # Compute moving average.
    smoothed = np.convolve(w, x, mode="same")
    smoothed = smoothed[size:-size]
    return smoothed
