# -*- coding: utf-8 -*-
import numpy as np

from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize
from ..signal import signal_resample
from ..signal import signal_interpolate



def ecg_rate(rpeaks, sampling_rate=1000, desired_length=None):
    """Calculate heart rate from R-peaks.

    Parameters
    ----------
    rpeaks : dict
        The samples at which the R-peak occur. Dict returned by
        `ecg_peaks()`.
    sampling_rate : int
        The sampling frequency of the signal that contains the R-peaks (in Hz,
        i.e., samples/second). Defaults to 1000Hz.
    desired_length : int
        By default, the returned heart rate has the same number of elements as
        peaks. If set to an integer, the returned heart rate will be
        interpolated between R-peaks over `desired_length` samples. Has no
        effect if a DataFrame is passed in as the `peaks` argument. Defaults to
        None.

    Returns
    -------
    array
        A Numpy array containing the heart rate.

    See Also
    --------
    ecg_clean, ecg_findpeaks, ecg_fixpeaks, ecg_process, ecg_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>> import matplotlib.pyplot as plt
    >>>
    >>> ecg = nk.ecg_simulate(duration=240, noise=0.1, heart_rate=70,
    >>>                       random_state=41)
    >>> rpeaks_uncorrected = nk.ecg_findpeaks(ecg)
    >>> artifacts, rpeaks_corrected = nk.ecg_fixpeaks(rpeaks_uncorrected,
    >>>                                               recursive=True,
    >>>                                               show=True)
    >>> rate_corrected = nk.ecg_rate(rpeaks_corrected,
    >>>                              desired_length=len(ecg))
    >>> rate_uncorrected = nk.ecg_rate(rpeaks_uncorrected,
                                       desired_length=len(ecg))
    >>>
    >>> fig, ax = plt.subplots()
    >>> ax.plot(rate_uncorrected, label="heart rate without artifact correction")
    >>> ax.plot(rate_corrected, label="heart rate with artifact correction")
    >>> ax.legend(loc="upper right")
    """
    # Get R-peaks indices from DataFrame or dict.
    rpeaks, _ = _signal_formatpeaks_sanitize(rpeaks, desired_length=None)

    rr = np.ediff1d(rpeaks, to_begin=0) / sampling_rate

    # The rate corresponding to the first peak is set to the mean RR.
    rr[0] = np.mean(rr)
    rate = 60 / rr

    if desired_length:
        rate = signal_interpolate(rpeaks, rate, desired_length=desired_length, method='quadratic')

#    if desired_length:
#        rate = signal_resample(rate, desired_length=desired_length,
#                               sampling_rate=sampling_rate)

    return rate
