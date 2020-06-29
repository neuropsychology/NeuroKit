# -*- coding: utf-8 -*-
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import pandas as pd

from ..stats import summary_plot
from .hrv_frequency import _hrv_frequency_show, hrv_frequency
from .hrv_nonlinear import _hrv_nonlinear_show, hrv_nonlinear
from .hrv_time import hrv_time
from .hrv_utils import _hrv_get_rri, _hrv_sanitize_input


def hrv(peaks, sampling_rate=1000, show=False):
    """Computes indices of Heart Rate Variability (HRV).

    Computes HRV indices in the time-, frequency-, and nonlinear domain. Note that a minimum duration
    of the signal containing the peaks is recommended for some HRV indices to be meaninful. For
    instance, 1, 2 and 5 minutes of high quality signal are the recomended minima for HF, LF and LF/HF,
    respectively. See references for details.

    Parameters
    ----------
    peaks : dict
        Samples at which cardiac extrema (i.e., R-peaks, systolic peaks) occur. Dictionary returned
        by ecg_findpeaks, ecg_peaks, ppg_findpeaks, or ppg_peaks.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks occur. Should be at
        least twice as high as the highest frequency in vhf. By default 1000.
    show : bool, optional
        If True, returns the plots that are generates for each of the domains.

    Returns
    -------
    DataFrame
        Contains HRV metrics from three domains:
        - frequency
        (see `hrv_frequency <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.hrv.hrv_frequency>`_)
        - time (see `hrv_time <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.hrv.hrv_time>`_)
        - non-linear
        (see `hrv_nonlinear <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.hrv.hrv_nonlinear`_)

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
    >>> hrv_indices = nk.hrv(peaks, sampling_rate=100, show=True)
    >>> hrv_indices #doctest: +SKIP

    References
    ----------
    - Stein, P. K. (2002). Assessing heart rate variability from real-world Holter reports. Cardiac
    electrophysiology review, 6(3), 239-244.

    - Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms.
    Frontiers in public health, 5, 258.

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
        # Indices for plotting
        out_plot = out.copy(deep=False)

        _hrv_plot(peaks, out_plot, sampling_rate)

    return out


def _hrv_plot(peaks, out, sampling_rate=1000):

    fig = plt.figure(constrained_layout=False)
    spec = gs.GridSpec(ncols=2, nrows=2, height_ratios=[1, 1], width_ratios=[1, 1])

    # Arrange grids
    ax_distrib = fig.add_subplot(spec[0, :-1])
    ax_distrib.set_xlabel("R-R intervals (ms)")
    ax_distrib.set_title("Distribution of R-R intervals")

    ax_psd = fig.add_subplot(spec[1, :-1])

    spec_within = gs.GridSpecFromSubplotSpec(4, 4, subplot_spec=spec[:, -1], wspace=0.025, hspace=0.05)
    ax_poincare = fig.add_subplot(spec_within[1:4, 0:3])
    ax_marg_x = fig.add_subplot(spec_within[0, 0:3])
    ax_marg_x.set_title("Poincaré Plot")
    ax_marg_y = fig.add_subplot(spec_within[1:4, 3])

    # Distribution of RR intervals
    peaks = _hrv_sanitize_input(peaks)
    rri = _hrv_get_rri(peaks, sampling_rate=sampling_rate, interpolate=False)
    ax_distrib = summary_plot(rri, ax=ax_distrib)

    # Poincare plot
    out.columns = [col.replace("HRV_", "") for col in out.columns]
    _hrv_nonlinear_show(rri, out, ax=ax_poincare, ax_marg_x=ax_marg_x, ax_marg_y=ax_marg_y)

    # PSD plot
    rri, sampling_rate = _hrv_get_rri(peaks, sampling_rate=sampling_rate, interpolate=True)
    frequency_bands = out[["ULF", "VLF", "LF", "HF", "VHF"]]
    _hrv_frequency_show(rri, frequency_bands, sampling_rate=sampling_rate, ax=ax_psd)
