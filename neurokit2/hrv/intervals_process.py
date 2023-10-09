# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np

from ..misc import NeuroKitWarning
from ..signal import signal_detrend, signal_interpolate
from .intervals_utils import (
    _intervals_sanitize,
    _intervals_time_to_sampling_rate,
    _intervals_time_uniform,
)


def intervals_process(
    intervals,
    intervals_time=None,
    interpolate=False,
    interpolation_rate=100,
    detrend=None,
    **kwargs
):
    """**Interval preprocessing**

    R-peak intervals preprocessing.

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
        high as the highest frequency in vhf. By default 100. To replicate Kubios defaults, set
        to 4.
    detrend : str
        Can be one of ``"polynomial"`` (traditional detrending of a given order) or
        ``"tarvainen2002"`` to use the smoothness priors approach described by Tarvainen (2002)
        (mostly used in HRV analyses as a lowpass filter to remove complex trends), ``"loess"`` for
        LOESS smoothing trend removal or ``"locreg"`` for local linear regression (the *'runline'*
        algorithm from chronux). By default None such that there is no detrending.
    **kwargs
        Keyword arguments to be passed to :func:`.signal_interpolate`.

    Returns
    -------
    np.ndarray
        Preprocessed intervals, in milliseconds.
    np.ndarray
        Preprocessed timestamps corresponding to intervals, in seconds.
    int
        Sampling rate (Hz) of the interpolated interbeat intervals.

    Examples
    --------
    **Example 1**: With interpolation and detrending
    .. ipython:: python

      import neurokit2 as nk
      import matplotlib.pyplot as plt

      plt.rc('font', size=8)

      # Download data
      data = nk.data("bio_resting_5min_100hz")
      sampling_rate = 100

      # Clean signal and find peaks
      ecg_cleaned = nk.ecg_clean(data["ECG"], sampling_rate=100)
      _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=100, correct_artifacts=True)
      peaks = info["ECG_R_Peaks"]

      # Convert peaks to intervals
      rri = np.diff(peaks) / sampling_rate * 1000
      rri_time = np.array(peaks[1:]) / sampling_rate

      # # Compute HRV indices
      # @savefig p_intervals_process1.png scale=100%
      # plt.figure()
      # plt.plot(intervals_time, intervals, label="Original intervals")
      # intervals, intervals_time = nk.intervals_process(rri,
      #                                               intervals_time=rri_time,
      #                                               interpolate=True,
      #                                               interpolation_rate=100,
      #                                               detrend="tarvainen2002")
      # plt.plot(intervals_time, intervals, label="Processed intervals")
      # plt.xlabel("Time (seconds)")
      # plt.ylabel("Interbeat intervals (milliseconds)")
      # @suppress
      # plt.close()

    """
    # Sanitize input
    intervals, intervals_time, _ = _intervals_sanitize(
        intervals, intervals_time=intervals_time
    )

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
            start=intervals_time[0],
            stop=intervals_time[-1] + 1 / interpolation_rate,
            step=1 / interpolation_rate,
        )

        intervals = signal_interpolate(intervals_time, intervals, x_new=x_new, **kwargs)
        intervals_time = x_new
    else:
        # check if intervals appear to be already interpolated
        if _intervals_time_uniform(intervals_time):
            # get sampling rate used for interpolation
            interpolation_rate = _intervals_time_to_sampling_rate(intervals_time)

    if detrend is not None:
        intervals = signal_detrend(
            intervals, method=detrend, sampling_rate=interpolation_rate
        )
    return intervals, intervals_time, interpolation_rate
