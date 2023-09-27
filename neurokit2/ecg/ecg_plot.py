# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from ..signal.signal_rate import _signal_rate_plot
from .ecg_peaks import _ecg_peaks_plot
from .ecg_segment import ecg_segment


def ecg_plot(ecg_signals, info=None):
    """**Visualize ECG data**

    Plot ECG signals and R-peaks.

    Parameters
    ----------
    ecg_signals : DataFrame
        DataFrame obtained from ``ecg_process()``.
    info : dict
        The information Dict returned by ``ecg_process()``. Defaults to ``None``.

    See Also
    --------
    ecg_process

    Returns
    -------
    Though the function returns nothing, the figure can be retrieved and saved as follows:

    .. code-block:: python

      # To be run after ecg_plot()
      fig = plt.gcf()
      fig.set_size_inches(10, 12, forward=True)
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
      nk.ecg_plot(signals, info)
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
        warn(
            "'info' dict not provided. Some information might be missing."
            + " Sampling rate will be set to 1000 Hz.",
            category=NeuroKitWarning,
        )
        info = {"sampling_rate": 1000}

    # Extract R-peaks (take those from df as it might have been cropped)
    if "ECG_R_Peaks" in ecg_signals.columns:
        info["ECG_R_Peaks"] = np.where(ecg_signals["ECG_R_Peaks"] == 1)[0]

    # Prepare figure and set axes.
    gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[2 / 3, 1 / 3])

    fig = plt.figure(constrained_layout=False)
    fig.suptitle("Electrocardiogram (ECG)", fontweight="bold")

    ax0 = fig.add_subplot(gs[0, :-1])
    ax1 = fig.add_subplot(gs[1, :-1], sharex=ax0)
    ax2 = fig.add_subplot(gs[:, -1])

    # Plot signals
    phase = None
    if "ECG_Phase_Ventricular" in ecg_signals.columns:
        phase = ecg_signals["ECG_Phase_Ventricular"].values

    ax0 = _ecg_peaks_plot(
        ecg_signals["ECG_Clean"].values,
        info=info,
        sampling_rate=info["sampling_rate"],
        raw=ecg_signals["ECG_Raw"].values,
        quality=ecg_signals["ECG_Quality"].values,
        phase=phase,
        ax=ax0,
    )

    # Plot Heart Rate
    ax1 = _signal_rate_plot(
        ecg_signals["ECG_Rate"].values,
        info["ECG_R_Peaks"],
        sampling_rate=info["sampling_rate"],
        title="Heart Rate",
        ytitle="Beats per minute (bpm)",
        color="#FF5722",
        color_mean="#FF9800",
        color_points="#FFC107",
        ax=ax1,
    )

    # Plot individual heart beats
    ax2 = ecg_segment(
        ecg_signals,
        info["ECG_R_Peaks"],
        info["sampling_rate"],
        show="return",
        ax=ax2,
    )
