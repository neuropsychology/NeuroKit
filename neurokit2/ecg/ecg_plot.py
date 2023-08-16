# -*- coding: utf-8 -*-
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..ecg import ecg_peaks
from ..epochs import epochs_to_df
from ..signal import signal_fixpeaks
from ..signal.signal_rate import _signal_rate_plot
from ..stats import rescale
from .ecg_segment import ecg_segment


def ecg_plot(ecg_signals, rpeaks=None, sampling_rate=1000, show_type="default"):
    """**Visualize ECG data**

    Plot ECG signals and R-peaks.

    Parameters
    ----------
    ecg_signals : DataFrame
        DataFrame obtained from ``ecg_process()``.
    rpeaks : dict
        The samples at which the R-peak occur. Dict returned by
        ``ecg_process()``. Defaults to ``None``.
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
      nk.ecg_plot(signals, sampling_rate=1000, show_type='default')
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
    peaks = np.where(ecg_signals["ECG_R_Peaks"] == 1)[0]

    # Prepare figure and set axes.
    if show_type in ["default", "full"]:
        x_axis = np.linspace(0, len(ecg_signals) / sampling_rate, len(ecg_signals))
        gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[2 / 3, 1 / 3])
        fig = plt.figure(constrained_layout=False)
        ax0 = fig.add_subplot(gs[0, :-1])
        ax0.set_xlabel("Time (seconds)")

        ax1 = fig.add_subplot(gs[1, :-1], sharex=ax0)
        ax2 = fig.add_subplot(gs[:, -1])

        fig.suptitle("Electrocardiogram (ECG)", fontweight="bold")
        plt.tight_layout(h_pad=0.3, w_pad=0.1)

        # Plot cleaned, raw ECG, R-peaks and signal quality.
        ax0.set_title("Raw and Cleaned Signal")

        quality = rescale(
            ecg_signals["ECG_Quality"],
            to=[np.min(ecg_signals["ECG_Clean"]), np.max(ecg_signals["ECG_Clean"])],
        )
        minimum_line = np.full(len(x_axis), quality.min())

        # Plot quality area first
        ax0.fill_between(
            x_axis,
            minimum_line,
            quality,
            alpha=0.12,
            zorder=0,
            interpolate=True,
            facecolor="#4CAF50",
            label="Quality",
        )

        # Plot signals
        ax0.plot(x_axis, ecg_signals["ECG_Raw"], color="#B0BEC5", label="Raw", zorder=1)
        ax0.plot(
            x_axis,
            ecg_signals["ECG_Clean"],
            color="#E91E63",
            label="Cleaned",
            zorder=1,
            linewidth=1.5,
        )
        ax0.scatter(
            x_axis[peaks],
            ecg_signals["ECG_Clean"][peaks],
            color="#FFC107",
            label="R-peaks",
            zorder=2,
        )

        # Optimize legend
        handles, labels = ax0.get_legend_handles_labels()
        order = [2, 0, 1, 3]
        ax0.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc="upper right",
        )

        # Plot Heart Rate
        ax1 = _signal_rate_plot(
            ecg_signals["ECG_Rate"].values,
            peaks,
            sampling_rate=sampling_rate,
            title="Heart Rate",
            ytitle="Beats per minute (bpm)",
            color="#FF5722",
            color_mean="#FF9800",
            color_points="red",
            ax=ax1,
        )

        # Plot individual heart beats
        ax2 = ecg_segment(
            ecg_signals["ECG_Clean"], peaks, sampling_rate, show="return", ax=ax2
        )

    # Plot artifacts
    if show_type in ["artifacts", "full"]:
        if sampling_rate is None:
            raise ValueError(
                "NeuroKit error: ecg_plot(): Sampling rate must be specified for artifacts"
                " to be plotted."
            )

        if rpeaks is None:
            _, rpeaks = ecg_peaks(ecg_signals["ECG_Clean"], sampling_rate=sampling_rate)

        fig = signal_fixpeaks(
            rpeaks,
            sampling_rate=sampling_rate,
            iterative=True,
            show=True,
            method="Kubios",
        )
