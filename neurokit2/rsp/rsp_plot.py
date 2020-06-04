# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rsp_plot(rsp_signals, sampling_rate=None):
    """Visualize respiration (RSP) data.

    Parameters
    ----------
    rsp_signals : DataFrame
        DataFrame obtained from `rsp_process()`.
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(duration=90, respiratory_rate=15)
    >>> rsp_signals, info = nk.rsp_process(rsp, sampling_rate=1000)
    >>> fig = nk.rsp_plot(rsp_signals)
    >>> fig #doctest: +SKIP

    Returns
    -------
    fig
        Figure representing a plot of the processed rsp signals.

    See Also
    --------
    rsp_process

    """
    # Mark peaks, troughs and phases.
    peaks = np.where(rsp_signals["RSP_Peaks"] == 1)[0]
    troughs = np.where(rsp_signals["RSP_Troughs"] == 1)[0]
    inhale = np.where(rsp_signals["RSP_Phase"] == 1)[0]
    exhale = np.where(rsp_signals["RSP_Phase"] == 0)[0]

    if "RSP_Amplitude" in list(rsp_signals.columns):
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)
    else:
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)

    # Determine unit of x-axis.
    last_ax = fig.get_axes()[-1]
    if sampling_rate is not None:
        last_ax.set_xlabel("Time (seconds)")
        x_axis = np.linspace(0, len(rsp_signals) / sampling_rate, len(rsp_signals))
    else:
        last_ax.set_xlabel("Samples")
        x_axis = np.arange(0, len(rsp_signals))

    plt.subplots_adjust(hspace=0.2)

    # Plot cleaned and raw respiration as well as peaks and troughs.
    ax0.set_title("Raw and Cleaned Signal")
    fig.suptitle("Respiration (RSP)", fontweight="bold")

    ax0.plot(x_axis, rsp_signals["RSP_Raw"], color="#B0BEC5", label="Raw", zorder=1)
    ax0.plot(x_axis, rsp_signals["RSP_Clean"], color="#2196F3", label="Cleaned", zorder=2, linewidth=1.5)

    ax0.scatter(x_axis[peaks], rsp_signals["RSP_Clean"][peaks], color="red", label="Inhalation Peaks", zorder=3)
    ax0.scatter(
        x_axis[troughs], rsp_signals["RSP_Clean"][troughs], color="orange", label="Exhalation Troughs", zorder=4
    )

    # Shade region to mark inspiration and expiration.
    exhale_signal, inhale_signal = _rsp_plot_phase(rsp_signals, troughs, peaks)

    ax0.fill_between(
        x_axis[exhale],
        exhale_signal[exhale],
        rsp_signals["RSP_Clean"][exhale],
        where=rsp_signals["RSP_Clean"][exhale] > exhale_signal[exhale],
        color="#CFD8DC",
        linestyle="None",
        label="exhalation",
    )
    ax0.fill_between(
        x_axis[inhale],
        inhale_signal[inhale],
        rsp_signals["RSP_Clean"][inhale],
        where=rsp_signals["RSP_Clean"][inhale] > inhale_signal[inhale],
        color="#ECEFF1",
        linestyle="None",
        label="inhalation",
    )

    ax0.legend(loc="upper right")

    # Plot rate and optionally amplitude.
    ax1.set_title("Breathing Rate")
    ax1.plot(x_axis, rsp_signals["RSP_Rate"], color="#4CAF50", label="Rate", linewidth=1.5)
    rate_mean = np.mean(rsp_signals["RSP_Rate"])
    ax1.axhline(y=rate_mean, label="Mean", linestyle="--", color="#4CAF50")
    ax1.legend(loc="upper right")

    if "RSP_Amplitude" in list(rsp_signals.columns):
        ax2.set_title("Breathing Amplitude")

        ax2.plot(x_axis, rsp_signals["RSP_Amplitude"], color="#009688", label="Amplitude", linewidth=1.5)
        amplitude_mean = np.mean(rsp_signals["RSP_Amplitude"])
        ax2.axhline(y=amplitude_mean, label="Mean", linestyle="--", color="#009688")
        ax2.legend(loc="upper right")

    plt.show()
    return fig


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
