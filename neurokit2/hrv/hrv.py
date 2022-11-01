# -*- coding: utf-8 -*-
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import pandas as pd

from ..stats import summary_plot
from .hrv_frequency import _hrv_frequency_show, hrv_frequency
from .hrv_nonlinear import _hrv_nonlinear_show, hrv_nonlinear
from .hrv_rsa import hrv_rsa
from .hrv_time import hrv_time
from .hrv_utils import _hrv_format_input
from .intervals_process import intervals_process


def hrv(peaks, sampling_rate=1000, show=False, **kwargs):
    """**Heart Rate Variability (HRV)**

    This function computes all HRV indices available in NeuroKit. It is essentially a convenience
    function that aggregates results from the :func:`time domain <hrv_time>`, :func:`frequency
    domain <hrv_frequency>`, and :func:`non-linear domain <hrv_nonlinear>`.

    .. tip::

        We strongly recommend checking our open-access paper `Pham et al. (2021)
        <https://doi.org/10.3390/s21123998>`_ on HRV indices as well as `Frasch (2022)
        <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9307944/>`_ for more information.

    Parameters
    ----------
    peaks : dict
        Samples at which R-peaks occur. Can be a list of indices or the output(s) of other
        functions such as :func:`.ecg_peaks`, :func:`.ppg_peaks`, :func:`.ecg_process` or
        :func:`bio_process`. Can also be a dict containing the keys `RRI` and `RRI_Time`
        to directly pass the R-R intervals and their timestamps, respectively.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks occur. Should be at
        least twice as high as the highest frequency in vhf. By default 1000.
    show : bool, optional
        If ``True``, returns the plots that are generated for each of the domains.

    Returns
    -------
    DataFrame
        Contains HRV indices in a DataFrame. If RSP data was provided (e.g., output of
        :func:`bio_process`), RSA indices will also be included.

    See Also
    --------
    hrv_time, hrv_frequency, hrv_nonlinear, hrv_rsa, .ecg_peaks, .ppg_peaks

    Examples
    --------
    **Example 1**: Only using a list of R-peaks locations

    .. ipython:: python

      import neurokit2 as nk
      import matplotlib.pyplot as plt
      plt.rc('font', size=8)

      # Download data
      data = nk.data("bio_resting_5min_100hz")

      # Clean signal and Find peaks
      ecg_cleaned = nk.ecg_clean(data["ECG"], sampling_rate=100)
      peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=100, correct_artifacts=True)

      # Compute HRV indices
      @savefig p_hrv1.png scale=100%
      hrv_indices = nk.hrv(peaks, sampling_rate=100, show=True)
      @suppress
      plt.close()

    **Example 2**: Compute HRV directly from processed data

    .. ipython:: python

      # Download data
      data = nk.data("bio_resting_5min_100hz")

      # Process
      signals, info = nk.bio_process(data, sampling_rate=100)

      # Get HRV
      nk.hrv(signals, sampling_rate=100)


    References
    ----------
    * Pham, T., Lau, Z. J., Chen, S. H. A., & Makowski, D. (2021). Heart Rate Variability in
      Psychology: A Review of HRV Indices and an Analysis Tutorial. Sensors, 21(12), 3998.
      https://doi.org/10.3390/s21123998
    * Frasch, M. G. (2022). Comprehensive HRV estimation pipeline in Python using Neurokit2:
      Application to sleep physiology. MethodsX, 9, 101782.
    * Stein, P. K. (2002). Assessing heart rate variability from real-world Holter reports. Cardiac
      electrophysiology review, 6(3), 239-244.
    * Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and
      norms. Frontiers in public health, 5, 258.

    """
    # Get indices
    out = []  # initialize empty container

    # Gather indices
    out.append(hrv_time(peaks, sampling_rate=sampling_rate))
    out.append(hrv_frequency(peaks, sampling_rate=sampling_rate))
    out.append(hrv_nonlinear(peaks, sampling_rate=sampling_rate))

    # Compute RSA if rsp data is available
    if isinstance(peaks, pd.DataFrame):
        if ("RSP_Phase" in peaks.columns) and ("RSP_Phase_Completion" in peaks.columns):
            rsp_signals = peaks[["RSP_Phase", "RSP_Phase_Completion"]]
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


# =============================================================================
# Plot
# =============================================================================
def _hrv_plot(peaks, out, sampling_rate=1000, interpolation_rate=100, **kwargs):

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

    plt.tight_layout(h_pad=0.5, w_pad=0.5)

    # Distribution of RR intervals
    rri, rri_time, rri_missing = _hrv_format_input(peaks, sampling_rate=sampling_rate)
    ax_distrib = summary_plot(rri, ax=ax_distrib, **kwargs)

    # Poincare plot
    out.columns = [col.replace("HRV_", "") for col in out.columns]
    _hrv_nonlinear_show(rri, rri_time=rri_time, rri_missing=rri_missing, out=out, ax=ax_poincare, ax_marg_x=ax_marg_x, ax_marg_y=ax_marg_y)

    # PSD plot
    rri, rri_time, sampling_rate = intervals_process(
        rri, intervals_time=rri_time, interpolate=True, interpolation_rate=interpolation_rate, **kwargs
    )

    frequency_bands = out[["ULF", "VLF", "LF", "HF", "VHF"]]
    _hrv_frequency_show(rri, frequency_bands, sampling_rate=sampling_rate, ax=ax_psd)
