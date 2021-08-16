# -*- coding: utf-8 -*-
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import pandas as pd

from ..stats import summary_plot
from .hrv_frequency import _hrv_frequency_show, hrv_frequency
from .hrv_nonlinear import _hrv_nonlinear_show, hrv_nonlinear
from .hrv_time import hrv_time
from .hrv_rsa import hrv_rsa
from .hrv_utils import _hrv_get_rri, _hrv_sanitize_input


def hrv(peaks, sampling_rate=1000, show=False, **kwargs):
    """Computes indices of Heart Rate Variability (HRV).

    Computes HRV indices in the time-, frequency-, and nonlinear domain. Note that a minimum duration
    of the signal containing the peaks is recommended for some HRV indices to be meaninful. For
    instance, 1, 2 and 5 minutes of high quality signal are the recomended minima for HF, LF and LF/HF,
    respectively. See references for details.

    Parameters
    ----------
    peaks : dict
        Samples at which cardiac extrema (i.e., R-peaks, systolic peaks) occur.
        Can be a list of indices or the output(s) of other functions such as ecg_peaks,
        ppg_peaks, ecg_process or bio_process.
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
        (see `hrv_nonlinear <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.hrv.hrv_nonlinear>`_)
        If RSP data is provided (e.g., output of `bio_process`):
        - rsa
         Otherwise, to compute ECG-derived respiration,
         use `hrv_rsa <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.hrv.hrv_rsa>`_
         If no raw respiratory data is available, users can also choose to use
         `ecg_rsp <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg.ecg_rsp`_ to
         obtain ECG-derived respiratory signal, although this is not an ideal procedure.

    See Also
    --------
    ecg_peaks, ppg_peaks, hrv_time, hrv_frequency, hrv_nonlinear, hrv_rsa

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Download data
    >>> data = nk.data("bio_resting_5min_100hz")
    >>>
    >>> # Clean signal and Find peaks
    >>> ecg_cleaned = nk.ecg_clean(data["ECG"], sampling_rate=100)
    >>> peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=100, correct_artifacts=True)
    >>>
    >>> # Compute HRV indices
    >>> hrv_indices = nk.hrv(peaks, sampling_rate=100, show=True)
    >>> hrv_indices #doctest: +SKIP
    >>>
    >>> # Compute HRV from processed signals
    >>> signals, info = nk.bio_process(data, sampling_rate=100)
    >>> hrv = nk.hrv(signals, sampling_rate=100, show=True)
    >>> hrv #doctest: +SKIP


    References
    ----------
    - Pham, T., Lau, Z. J., Chen, S. H. A., & Makowski, D. (2021). Heart Rate Variability in Psychology:
    A Review of HRV Indices and an Analysis Tutorial. Sensors, 21(12), 3998. https://doi:10.3390/s21123998

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

    # Compute RSA if rsp data is available
    if isinstance(peaks, pd.DataFrame):
        rsp_cols = [col for col in peaks.columns if "RSP_Phase" in col]
        if len(rsp_cols) == 2:
            rsp_signals = peaks[rsp_cols]
            rsa = hrv_rsa(peaks, rsp_signals, sampling_rate=sampling_rate)
            out.append(pd.DataFrame([rsa]))

    out = pd.concat(out, axis=1)

    # Plot
    if show:
        if isinstance(peaks, dict):
            peaks = peaks["ECG_R_Peaks"]
        # Indices for plotting
        out_plot = out.copy(deep=False)

        _hrv_plot(peaks, out_plot, sampling_rate, **kwargs)

    return out


def _hrv_plot(peaks, out, sampling_rate=1000, **kwargs):

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
    ax_distrib = summary_plot(rri, ax=ax_distrib, **kwargs)

    # Poincare plot
    out.columns = [col.replace("HRV_", "") for col in out.columns]
    _hrv_nonlinear_show(rri, out, ax=ax_poincare, ax_marg_x=ax_marg_x, ax_marg_y=ax_marg_y)

    # PSD plot
    rri, sampling_rate = _hrv_get_rri(peaks, sampling_rate=sampling_rate, interpolate=True)
    frequency_bands = out[["ULF", "VLF", "LF", "HF", "VHF"]]
    _hrv_frequency_show(rri, frequency_bands, sampling_rate=sampling_rate, ax=ax_psd)
