# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np

from ..misc import NeuroKitWarning
from .signal_formatpeaks import _signal_formatpeaks_sanitize
from .signal_interpolate import signal_interpolate


def signal_period(
    peaks,
    sampling_rate=1000,
    desired_length=None,
    interpolation_method="monotone_cubic",
):
    """**Calculate signal period from a series of peaks**

    Calculate the period of a signal from a series of peaks. The period is defined as the time
    in seconds between two consecutive peaks.

    Parameters
    ----------
    peaks : Union[list, np.array, pd.DataFrame, pd.Series, dict]
        The samples at which the peaks occur. If an array is passed in, it is assumed that it was
        obtained with :func:`.signal_findpeaks`. If a DataFrame is passed in, it is assumed it is
        of the same length as the input signal in which occurrences of R-peaks are marked as "1",
        with such containers obtained with e.g., :func:`.ecg_findpeaks` or :func:`.rsp_findpeaks`.
    sampling_rate : int
        The sampling frequency of the signal that contains peaks (in Hz, i.e., samples/second).
        Defaults to 1000.
    desired_length : int
        If left at the default ``None``, the returned period will have the same number of elements
        as ``peaks``. If set to a value larger than the sample at which the last peak occurs in the
        signal (i.e., ``peaks[-1]``), the returned period will be interpolated between peaks over
        ``desired_length`` samples. To interpolate the period over the entire duration of the
        signal, set ``desired_length`` to the number of samples in the signal. Cannot be smaller
        than or equal to the sample at which the last peak occurs in the signal.
        Defaults to ``None``.
    interpolation_method : str
        Method used to interpolate the rate between peaks. See :func:`.signal_interpolate`.
        ``"monotone_cubic"`` is chosen as the default interpolation method since it ensures monotone
        interpolation between data points (i.e., it prevents physiologically implausible
        "overshoots" or "undershoots" in the y-direction). In contrast, the widely used cubic
        spline interpolation does not ensure monotonicity.

    Returns
    -------
    array
        A vector containing the period.

    See Also
    --------
    signal_findpeaks, signal_fixpeaks, signal_plot

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Generate 2 signals (with fixed and variable period)
      sig1 = nk.signal_simulate(duration=20, sampling_rate=200, frequency=1)
      sig2 = nk.ecg_simulate(duration=20, sampling_rate=200, heart_rate=60)

      # Find peaks
      info1 = nk.signal_findpeaks(sig1)
      info2 = nk.ecg_findpeaks(sig2, sampling_rate=200)

      # Compute period
      period1 = nk.signal_period(peaks=info1["Peaks"], desired_length=len(sig1), sampling_rate=200)
      period2 = nk.signal_period(peaks=info2["ECG_R_Peaks"], desired_length=len(sig2), sampling_rate=200)

      @savefig p_signal_period.png scale=100%
      nk.signal_plot([period1, period2], subplots=True)
      @suppress
      plt.close()

    """
    peaks = _signal_formatpeaks_sanitize(peaks)

    # Sanity checks.
    if np.size(peaks) <= 3:
        warn(
            "Too few peaks detected to compute the rate. Returning empty vector.",
            category=NeuroKitWarning,
        )
        return np.full(desired_length, np.nan)

    if isinstance(desired_length, (int, float)):
        if desired_length <= peaks[-1]:
            raise ValueError(
                "NeuroKit error: desired_length must be None or larger than the index of the last peak."
            )

    # Calculate period in sec, based on peak to peak difference and make sure
    # that rate has the same number of elements as peaks (important for
    # interpolation later) by prepending the mean of all periods.
    period = np.ediff1d(peaks, to_begin=0) / sampling_rate
    period[0] = np.mean(period[1:])

    # Interpolate all statistics to desired length.
    if desired_length is not None:
        period = signal_interpolate(
            peaks, period, x_new=np.arange(desired_length), method=interpolation_method
        )

    return period
