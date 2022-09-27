# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np

from ..signal import signal_interpolate
from .hrv_utils import intervals_sanitize

from ..misc import NeuroKitWarning


def intervals_preprocess(intervals, intervals_time=None, interpolate=False, interpolation_rate=100, **kwargs):
    intervals, intervals_time = intervals_sanitize(intervals, intervals_time=intervals_time)

    if interpolate is False:
        interpolation_rate = None

    else:
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
            start=intervals_time[0],
            stop=intervals_time[-1] + 1 / interpolation_rate,
            step=1 / interpolation_rate,
        )

        intervals = signal_interpolate(intervals_time, intervals, x_new=x_new, **kwargs)
    return intervals, interpolation_rate
