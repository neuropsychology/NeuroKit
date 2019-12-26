# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal


def signal_smoothing(signal, kernel='boxzen', size=10):
    """Signal smoothing.

    This implementation uses the convolution of a filter kernel with the input signal to compute the smoothed signal (Smith, 1997).

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    kernel : str, array, optional
        Type of kernel to use; if array, use directly as the kernel. Can be one of 'median', 'boxzen', 'boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann', 'kaiser' (needs beta), 'gaussian' (needs std), 'general_gaussian' (needs power, width), 'slepian' (needs width) or 'chebwin' (needs attenuation).
    size : int, optional
        Size of the kernel; ignored if kernel is an array.

    Returns
    -------
    signal : array
        Smoothed signal.
    params : dict
        Smoother parameters.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=10, num=1000))
    >>> distorted = nk.signal_distord(signal, noise_amplitude=[0.3, 0.2, 0.1], noise_frequency=[5, 10, 50])
    >>>
    >>> size = len(signal)/100
    >>> signals = pd.DataFrame({
            "Raw": distorted,
            "Median": nk.signal_smoothing(distorted, kernel='median', size=size-1),
            "BoxZen": nk.signal_smoothing(distorted, kernel='boxzen', size=size),
            "Triang": nk.signal_smoothing(distorted, kernel='triang', size=size),
            "Blackman": nk.signal_smoothing(distorted, kernel='blackman', size=size)})
    >>> signals.plot()

    References
    ----------
    - Smith, S. W. (1997). The scientist and engineer's guide to digital signal processing.
    """
    length = len(signal)

    if isinstance(kernel, str) is False:
        raise TypeError("NeuroKit error: signal_smoothing(): 'kernel' "
                         "should be a string.")


    # check length
    if size > length or size < 1:
        raise TypeError("NeuroKit error: signal_smoothing(): 'size' "
                         "should be between 1 and length of the signal.")

    if kernel == 'boxzen':
        # hybrid method
        # 1st pass - boxcar kernel
        x = _signal_smoothing(signal, kernel='boxcar', size=size)

        # 2nd pass - parzen kernel
        smoothed = _signal_smoothing(x, kernel='parzen', size=size)

    elif kernel == 'median':
        smoothed = _signal_smoothing_median(signal, size)

    else:
        smoothed = _signal_smoothing(signal, kernel=kernel, size=size)

    return smoothed





# =============================================================================
# Internals
# =============================================================================
def _signal_smoothing_median(signal, size=5):
    # median filter
    if size % 2 == 0:
        raise ValueError("NeuroKit error: signal_smoothing(): When the "
                         "kernel is 'median', 'size' must be odd.")

    smoothed = scipy.signal.medfilt(signal, kernel_size=int(size))

    return smoothed




def _signal_smoothing(signal, kernel="boxcar", size=5):

    # Get window
    size = int(size)
    window = scipy.signal.get_window(kernel, size)
    w = window / window.sum()

    # Extend signal edges to avoid boundary effects
    x = np.concatenate((signal[0] * np.ones(size), signal, signal[-1] * np.ones(size)))

    # convolve
    smoothed = np.convolve(w, x, mode='same')
    smoothed = smoothed[size:-size]
    return smoothed





