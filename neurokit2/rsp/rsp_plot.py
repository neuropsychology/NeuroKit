# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rsp_plot(rsp_signals, sampling_rate=None, figsize=(10, 10)):
    """**Visualize respiration (RSP) data**

    Parameters
    ----------
    rsp_signals : DataFrame
        DataFrame obtained from :func:`.rsp_process`.
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).
    figsize : tuple
        The size of the figure (width, height) in inches.

    See Also
    --------
    rsp_process

    Returns
    -------
    Though the function returns nothing, the figure can be retrieved and saved as follows:

    .. code-block:: console

        # To be run after rsp_plot()
        fig = plt.gcf()
        fig.savefig("myfig.png")

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate data
      rsp = nk.rsp_simulate(duration=90, respiratory_rate=15)

      # Process signal
      rsp_signals, info = nk.rsp_process(rsp, sampling_rate=1000)

      # Plot
      @savefig p_rsp_plot1.png scale=100%
      nk.rsp_plot(rsp_signals, sampling_rate=1000)
      @suppress
      plt.close()

    """
    # Mark peaks, troughs and phases.
    peaks = np.where(rsp_signals["RSP_Peaks"] == 1)[0]
    troughs = np.where(rsp_signals["RSP_Troughs"] == 1)[0]
    inhale = np.where(rsp_signals["RSP_Phase"] == 1)[0]
    exhale = np.where(rsp_signals["RSP_Phase"] == 0)[0]

    nrow = 2
    if "RSP_Amplitude" in list(rsp_signals.columns):
        nrow += 1
    if "RSP_RVT" in list(rsp_signals.columns):
        nrow += 1
    if "RSP_Symmetry_PeakTrough" in list(rsp_signals.columns):
        nrow += 1

    fig, ax = plt.subplots(nrows=nrow, ncols=1, sharex=True, figsize=figsize)

    # Determine unit of x-axis.
    last_ax = fig.get_axes()[-1]
    if sampling_rate is not None:
        last_ax.set_xlabel("Time (seconds)")
        x_axis = np.linspace(0, len(rsp_signals) / sampling_rate, len(rsp_signals))
    else:
        last_ax.set_xlabel("Samples")
        x_axis = np.arange(0, len(rsp_signals))

    # Plot cleaned and raw respiration as well as peaks and troughs.
    ax[0].set_title("Raw and Cleaned Signal")
    fig.suptitle("Respiration (RSP)", fontweight="bold")

    ax[0].plot(x_axis, rsp_signals["RSP_Raw"], color="#B0BEC5", label="Raw", zorder=1)
    ax[0].plot(
        x_axis, rsp_signals["RSP_Clean"], color="#2196F3", label="Cleaned", zorder=2, linewidth=1.5
    )

    ax[0].scatter(
        x_axis[peaks],
        rsp_signals["RSP_Clean"][peaks],
        color="red",
        label="Exhalation Onsets",
        zorder=3,
    )
    ax[0].scatter(
        x_axis[troughs],
        rsp_signals["RSP_Clean"][troughs],
        color="orange",
        label="Inhalation Onsets",
        zorder=4,
    )

    # Shade region to mark inspiration and expiration.
    exhale_signal, inhale_signal = _rsp_plot_phase(rsp_signals, troughs, peaks)

    ax[0].fill_between(
        x_axis[exhale],
        exhale_signal[exhale],
        rsp_signals["RSP_Clean"][exhale],
        where=rsp_signals["RSP_Clean"][exhale] > exhale_signal[exhale],
        color="#CFD8DC",
        linestyle="None",
        label="exhalation",
    )
    ax[0].fill_between(
        x_axis[inhale],
        inhale_signal[inhale],
        rsp_signals["RSP_Clean"][inhale],
        where=rsp_signals["RSP_Clean"][inhale] > inhale_signal[inhale],
        color="#ECEFF1",
        linestyle="None",
        label="inhalation",
    )

    ax[0].legend(loc="upper right")

    # Plot rate and optionally amplitude.
    ax[1].set_title("Breathing Rate")
    ax[1].plot(x_axis, rsp_signals["RSP_Rate"], color="#4CAF50", label="Rate", linewidth=1.5)
    rate_mean = np.mean(rsp_signals["RSP_Rate"])
    ax[1].axhline(y=rate_mean, label="Mean", linestyle="--", color="#4CAF50")
    ax[1].legend(loc="upper right")

    if "RSP_Amplitude" in list(rsp_signals.columns):
        ax[2].set_title("Breathing Amplitude")

        ax[2].plot(
            x_axis, rsp_signals["RSP_Amplitude"], color="#009688", label="Amplitude", linewidth=1.5
        )
        amplitude_mean = np.mean(rsp_signals["RSP_Amplitude"])
        ax[2].axhline(y=amplitude_mean, label="Mean", linestyle="--", color="#009688")
        ax[2].legend(loc="upper right")

    if "RSP_RVT" in list(rsp_signals.columns):
        ax[3].set_title("Respiratory Volume per Time")

        ax[3].plot(x_axis, rsp_signals["RSP_RVT"], color="#00BCD4", label="RVT", linewidth=1.5)
        rvt_mean = np.mean(rsp_signals["RSP_RVT"])
        ax[3].axhline(y=rvt_mean, label="Mean", linestyle="--", color="#009688")
        ax[3].legend(loc="upper right")

    if "RSP_Symmetry_PeakTrough" in list(rsp_signals.columns):
        ax[4].set_title("Cycle Symmetry")

        ax[4].plot(
            x_axis,
            rsp_signals["RSP_Symmetry_PeakTrough"],
            color="green",
            label="Peak-Trough Symmetry",
            linewidth=1.5,
        )
        ax[4].plot(
            x_axis,
            rsp_signals["RSP_Symmetry_RiseDecay"],
            color="purple",
            label="Rise-Decay Symmetry",
            linewidth=1.5,
        )
        ax[4].legend(loc="upper right")


# =============================================================================
# Internals
# =============================================================================
def _rsp_plot_phase(rsp_signals, troughs, peaks):

    exhale_signal = pd.Series(np.full(len(rsp_signals), np.nan))
    exhale_signal[troughs] = rsp_signals["RSP_Clean"][troughs].values
    exhale_signal[peaks] = rsp_signals["RSP_Clean"][peaks].values
    exhale_signal = exhale_signal.fillna(method="backfill")

    inhale_signal = pd.Series(np.full(len(rsp_signals), np.nan))
    inhale_signal[troughs] = rsp_signals["RSP_Clean"][troughs].values
    inhale_signal[peaks] = rsp_signals["RSP_Clean"][peaks].values
    inhale_signal = inhale_signal.fillna(method="ffill")

    return exhale_signal, inhale_signal
