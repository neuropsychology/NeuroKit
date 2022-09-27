# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np

from ..misc import NeuroKitWarning
from ..signal import signal_interpolate
from .intervals_utils import _intervals_sanitize, _intervals_time_to_sampling_rate, _intervals_time_uniform


def intervals_preprocess(intervals, intervals_time=None, interpolate=False, interpolation_rate=100, **kwargs):
    """**Interval preprocessing**

    Parameters
    ----------
    intervals : list or array
        List or numpy array of intervals, in milliseconds.
    intervals_time : list or array, optional
        List or numpy array of timestamps corresponding to intervals, in seconds.
    interpolate : bool, optional
        Whether to interpolate the interval signal. The default is False.
    interpolation_rate : int, optional
        Sampling rate (Hz) of the interpolated interbeat intervals. Should be at least twice as
        high as the highest frequency in vhf. By default 100. To replicate Kubios defaults, set to 4.
    **kwargs
        Keyword arguments to be passed to :func:`.signal_interpolate`.

    Returns
    -------
    intervals : array
        Preprocessed intervals, in milliseconds.
    intervals_time : array
        Preprocessed timestamps corresponding to intervals, in seconds.

    """
    intervals, intervals_time = _intervals_sanitize(intervals, intervals_time=intervals_time)

    if interpolate is False:
        interpolation_rate = None

    if interpolation_rate is not None:
        # Rate should be at least 1 Hz (due to Nyquist & frequencies we are interested in)
        # We considered an interpolation rate 4 Hz by default to match Kubios
        # but in case of some applications with high heart rates we decided to make it 100 Hz
        # See https://github.com/neuropsychology/NeuroKit/pull/680 for more information
        # and if you have any thoughts to contribute, please let us know!
        if interpolation_rate < 1:
            warn(
                "The interpolation rate of the R-R intervals is too low for "
                " computing the frequency-domain features."
                " Consider increasing the interpolation rate to at least 1 Hz.",
                category=NeuroKitWarning,
            )

        # Compute x-values of interpolated interval signal at requested sampling rate.
        x_new = np.arange(
            start=intervals_time[0], stop=intervals_time[-1] + 1 / interpolation_rate, step=1 / interpolation_rate,
        )

        intervals = signal_interpolate(intervals_time, intervals, x_new=x_new, **kwargs)
    else:
        # check if intervals appear to be already interpolated
        if _intervals_time_uniform(intervals_time):
            # get sampling rate used for interpolation
            interpolation_rate = _intervals_time_to_sampling_rate(intervals_time)
    return intervals, interpolation_rate
