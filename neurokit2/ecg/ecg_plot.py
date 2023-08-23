# -*- coding: utf-8 -*-
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..signal.signal_rate import _signal_rate_plot
from .ecg_peaks import _ecg_peaks_plot
from .ecg_segment import ecg_segment


def ecg_plot(ecg_signals, info=None, sampling_rate=1000, show_type="default"):
    """**Visualize ECG data**

    Plot ECG signals and R-peaks.

    Parameters
    ----------
    ecg_signals : DataFrame
        DataFrame obtained from ``ecg_process()``.
    info : dict
        The information Dict returned by ``ecg_process()``. Defaults to ``None``.
    sampling_rate : int
        The sampling frequency of ``ecg_cleaned`` (in Hz, i.e., samples/second). Defaults to 1000.
    show_type : str
        Visualize the ECG data with ``"default"`` or visualize artifacts thresholds with
        ``"artifacts"`` produced by ``ecg_fixpeaks()``, or ``"full"`` to visualize both.

    See Also
    --------
    ecg_process

    Returns
    -------
    Though the function returns nothing, the figure can be retrieved and saved as follows:

    .. code-block:: console

        # To be run after ecg_plot()
        fig = plt.gcf()
        fig.savefig("myfig.png")

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate data
      ecg = nk.ecg_simulate(duration=15, sampling_rate=1000, heart_rate=80)

      # Process signal
      signals, info = nk.ecg_process(ecg, sampling_rate=1000)

      # Plot
      @savefig p_ecg_plot.png scale=100%
      nk.ecg_plot(signals, info, sampling_rate=1000, show_type='default')
      @suppress
      plt.close()

    """
    # Sanity-check input.
    if not isinstance(ecg_signals, pd.DataFrame):
        raise ValueError(
            "NeuroKit error: ecg_plot(): The `ecg_signals` argument must be the "
            "DataFrame returned by `ecg_process()`."
        )

    # Extract R-peaks.
    if info is None:
        info = {"ECG_R_Peaks": np.where(ecg_signals["ECG_R_Peaks"] == 1)[0]}

    # Prepare figure and set axes.
    gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[2 / 3, 1 / 3])

    fig = plt.figure(constrained_layout=False)
    fig.suptitle("Electrocardiogram (ECG)", fontweight="bold")

    ax0 = fig.add_subplot(gs[0, :-1])
    ax1 = fig.add_subplot(gs[1, :-1], sharex=ax0)
    ax2 = fig.add_subplot(gs[:, -1])

    # Plot signals
    ax0 = _ecg_peaks_plot(
        ecg_signals["ECG_Clean"].values,
        info=info,
        sampling_rate=sampling_rate,
        raw=ecg_signals["ECG_Raw"].values,
        quality=ecg_signals["ECG_Quality"].values,
        ax=ax0,
    )

    # Plot Heart Rate
    ax1 = _signal_rate_plot(
        ecg_signals["ECG_Rate"].values,
        info["ECG_R_Peaks"],
        sampling_rate=sampling_rate,
        title="Heart Rate",
        ytitle="Beats per minute (bpm)",
        color="#FF5722",
        color_mean="#FF9800",
        color_points="#FFC107",
        ax=ax1,
    )

    # Plot individual heart beats
    ax2 = ecg_segment(
        ecg_signals["ECG_Clean"],
        info["ECG_R_Peaks"],
        sampling_rate,
        show="return",
        ax=ax2,
    )
