# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from .hrv_time import hrv_time
from .hrv_frequency import hrv_frequency
from .hrv_frequency import _hrv_frequency_show
from .hrv_nonlinear import hrv_nonlinear
from .hrv_nonlinear import _hrv_nonlinear_show
from .hrv_utils import _hrv_get_rri
from .hrv_utils import _hrv_sanitize_input
from ..stats import summary_plot


def hrv(peaks, sampling_rate=1000, show=False):
    """ Computes indices of Heart Rate Variability (HRV).

    Computes HRV indices in the time-, frequency-, and nonlinear domain. Note
    that a minimum duration of the signal containing the peaks is recommended
    for some HRV indices to be meaninful. For instance, 1, 2 and 5 minutes of
    high quality signal are the recomended minima for HF, LF and LF/HF,
    respectively. See references for details.

    Parameters
    ----------
    peaks : dict
        Samples at which cardiac extrema (i.e., R-peaks, systolic peaks) occur.
        Dictionary returned by ecg_findpeaks, ecg_peaks, ppg_findpeaks, or
        ppg_peaks.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks
        occur. Should be at least twice as high as the highest frequency in vhf.
        By default 1000.
    show : bool, optional
        If True, returns the plots that are generates for each of the domains.

    Returns
    -------
    DataFrame
        Contains HRV metrics from three domains:
        - frequency (for details see hrv_frequency)
        - time (for details see hrv_time)
        - non-linear (for details see hrv_nonlinear)
    See Also
    --------
    ecg_peaks, ppg_peaks, hrv_time, hrv_frequency, hrv_nonlinear

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Download data
    >>> data = nk.data("bio_resting_5min_100hz")
    >>>
    >>> # Find peaks
    >>> peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)
    >>>
    >>> # Compute HRV indices
    >>> nk.hrv(peaks, sampling_rate=100, show=True)

    References
    ----------
    - Stein, P. K. (2002). Assessing heart rate variability from real-world
      Holter reports. Cardiac electrophysiology review, 6(3), 239-244.
    - Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate
    variability metrics and norms. Frontiers in public health, 5, 258.
    """
    # Get indices
    out = []  # initialize empty container

    # Gather indices
    out.append(hrv_time(peaks, sampling_rate=sampling_rate))
    out.append(hrv_frequency(peaks, sampling_rate=sampling_rate))
    out.append(hrv_nonlinear(peaks, sampling_rate=sampling_rate))

    out = pd.concat(out, axis=1)

    # Plot
    if show:
        _hrv_plot(peaks, sampling_rate)

    return out




def _hrv_plot(peaks, sampling_rate=1000):
    fig = plt.figure(constrained_layout=False)
    spec = matplotlib.gridspec.GridSpec(ncols=2, nrows=2,
                                        height_ratios=[1, 1], width_ratios=[1, 1])

    # Arrange grids
    ax_distrib = fig.add_subplot(spec[0, :-1])
    ax_distrib.set_xlabel('R-R intervals (ms)')
    ax_distrib.set_title("Distribution of R-R intervals")

    ax_psd = fig.add_subplot(spec[1, :-1])
    ax_poincare = fig.add_subplot(spec[:, -1])

    # Distribution of RR intervals
    peaks = _hrv_sanitize_input(peaks)
    rri = _hrv_get_rri(peaks, sampling_rate=sampling_rate, interpolate=False)
    ax_distrib = summary_plot(rri, ax=ax_distrib)

    # Poincare plot
    out_poincare = out.copy()
    out_poincare.columns = [col.replace('HRV_', '') for col in out_poincare.columns]
    ax_poincare = _hrv_nonlinear_show(rri, out_poincare, ax=ax_poincare)

    # PSD plot
    rri, sampling_rate = _hrv_get_rri(peaks,
                                      sampling_rate=sampling_rate, interpolate=True)
    _hrv_frequency_show(rri, out_poincare,
                        sampling_rate=sampling_rate, ax=ax_psd)
