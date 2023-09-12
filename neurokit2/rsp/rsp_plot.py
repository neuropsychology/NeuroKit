# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning


def rsp_plot(rsp_signals, info=None, static=True):
    """**Visualize respiration (RSP) data**

    Parameters
    ----------
    rsp_signals : DataFrame
        DataFrame obtained from :func:`.rsp_process`.
    info : dict
        The information Dict returned by ``rsp_process()``. Defaults to ``None``.
    static : bool
        If True, a static plot will be generated with matplotlib.
        If False, an interactive plot will be generated with plotly.
        Defaults to True.

    See Also
    --------
    rsp_process

    Returns
    -------
    See :func:`.ecg_plot` for details on how to access the figure, modify the size and save it.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate data
      rsp = nk.rsp_simulate(duration=90, respiratory_rate=15, sampling_rate=100)

      # Process signal
      rsp_signals, info = nk.rsp_process(rsp, sampling_rate=100)

      # Plot
      @savefig p_rsp_plot1.png scale=100%
      nk.rsp_plot(rsp_signals, info)
      @suppress
      plt.close()

    """
    if info is None:
        warn(
            "'info' dict not provided. Some information might be missing."
            + " Sampling rate will be set to 1000 Hz.",
            category=NeuroKitWarning,
        )

        info = {
            "sampling_rate": 1000,
        }

    # Mark peaks, troughs and phases.
    peaks = np.where(rsp_signals["RSP_Peaks"] == 1)[0]
    troughs = np.where(rsp_signals["RSP_Troughs"] == 1)[0]
    inhale = np.where(rsp_signals["RSP_Phase"] == 1)[0]
    exhale = np.where(rsp_signals["RSP_Phase"] == 0)[0]

    nrow = 2

    # Determine mean rate.
    rate_mean = np.mean(rsp_signals["RSP_Rate"])

    if "RSP_Amplitude" in list(rsp_signals.columns):
        nrow += 1
        # Determine mean amplitude.
        amplitude_mean = np.mean(rsp_signals["RSP_Amplitude"])
    if "RSP_RVT" in list(rsp_signals.columns):
        nrow += 1
        # Determine mean RVT.
        rvt_mean = np.mean(rsp_signals["RSP_RVT"])
    if "RSP_Symmetry_PeakTrough" in list(rsp_signals.columns):
        nrow += 1

    # Get signals marking inspiration and expiration.
    exhale_signal, inhale_signal = _rsp_plot_phase(rsp_signals, troughs, peaks)

    # Determine unit of x-axis.
    x_label = "Time (seconds)"
    x_axis = np.linspace(0, len(rsp_signals) / info["sampling_rate"], len(rsp_signals))

    if static:
        fig, ax = plt.subplots(nrows=nrow, ncols=1, sharex=True)

        last_ax = fig.get_axes()[-1]
        last_ax.set_xlabel(x_label)

        # Plot cleaned and raw respiration as well as peaks and troughs.
        ax[0].set_title("Raw and Cleaned Signal")
        fig.suptitle("Respiration (RSP)", fontweight="bold")

        ax[0].plot(
            x_axis, rsp_signals["RSP_Raw"], color="#B0BEC5", label="Raw", zorder=1
        )
        ax[0].plot(
            x_axis,
            rsp_signals["RSP_Clean"],
            color="#2196F3",
            label="Cleaned",
            zorder=2,
            linewidth=1.5,
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
        ax[1].plot(
            x_axis,
            rsp_signals["RSP_Rate"],
            color="#4CAF50",
            label="Rate",
            linewidth=1.5,
        )
        ax[1].axhline(y=rate_mean, label="Mean", linestyle="--", color="#4CAF50")
        ax[1].legend(loc="upper right")

        if "RSP_Amplitude" in list(rsp_signals.columns):
            ax[2].set_title("Breathing Amplitude")

            ax[2].plot(
                x_axis,
                rsp_signals["RSP_Amplitude"],
                color="#009688",
                label="Amplitude",
                linewidth=1.5,
            )
            ax[2].axhline(
                y=amplitude_mean, label="Mean", linestyle="--", color="#009688"
            )
            ax[2].legend(loc="upper right")

        if "RSP_RVT" in list(rsp_signals.columns):
            ax[3].set_title("Respiratory Volume per Time")

            ax[3].plot(
                x_axis,
                rsp_signals["RSP_RVT"],
                color="#00BCD4",
                label="RVT",
                linewidth=1.5,
            )
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
    else:
        # Generate interactive plot with plotly.
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

        except ImportError as e:
            raise ImportError(
                "NeuroKit error: rsp_plot(): the 'plotly'",
                " module is required when 'static' is False.",
                " Please install it first (`pip install plotly`).",
            ) from e

        subplot_titles = ["Raw and Cleaned Signal", "Breathing Rate"]
        if "RSP_Amplitude" in list(rsp_signals.columns):
            subplot_titles.append("Breathing Amplitude")
        if "RSP_RVT" in list(rsp_signals.columns):
            subplot_titles.append("Respiratory Volume per Time")
        if "RSP_Symmetry_PeakTrough" in list(rsp_signals.columns):
            subplot_titles.append("Cycle Symmetry")
        subplot_titles = tuple(subplot_titles)
        fig = make_subplots(
            rows=nrow,
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
        )

        # Plot cleaned and raw RSP
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=rsp_signals["RSP_Raw"], name="Raw", marker_color="#B0BEC5"
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=rsp_signals["RSP_Clean"],
                name="Cleaned",
                marker_color="#2196F3",
            ),
            row=1,
            col=1,
        )

        # Plot peaks and troughs.
        fig.add_trace(
            go.Scatter(
                x=x_axis[peaks],
                y=rsp_signals["RSP_Clean"][peaks],
                name="Exhalation Onsets",
                marker_color="red",
                mode="markers",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis[troughs],
                y=rsp_signals["RSP_Clean"][troughs],
                name="Inhalation Onsets",
                marker_color="orange",
                mode="markers",
            ),
            row=1,
            col=1,
        )

        # TODO: Shade region to mark inspiration and expiration.

        # Plot rate and optionally amplitude.
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=rsp_signals["RSP_Rate"], name="Rate", marker_color="#4CAF50"
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=[rate_mean] * len(x_axis),
                name="Mean Rate",
                marker_color="#4CAF50",
                line=dict(dash="dash"),
            ),
            row=2,
            col=1,
        )

        if "RSP_Amplitude" in list(rsp_signals.columns):
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=rsp_signals["RSP_Amplitude"],
                    name="Amplitude",
                    marker_color="#009688",
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=[amplitude_mean] * len(x_axis),
                    name="Mean Amplitude",
                    marker_color="#009688",
                    line=dict(dash="dash"),
                ),
                row=3,
                col=1,
            )

        if "RSP_RVT" in list(rsp_signals.columns):
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=rsp_signals["RSP_RVT"],
                    name="RVT",
                    marker_color="#00BCD4",
                ),
                row=4,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=[rvt_mean] * len(x_axis),
                    name="Mean RVT",
                    marker_color="#00BCD4",
                    line=dict(dash="dash"),
                ),
                row=4,
                col=1,
            )

        if "RSP_Symmetry_PeakTrough" in list(rsp_signals.columns):
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=rsp_signals["RSP_Symmetry_PeakTrough"],
                    name="Peak-Trough Symmetry",
                    marker_color="green",
                ),
                row=5,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=rsp_signals["RSP_Symmetry_RiseDecay"],
                    name="Rise-Decay Symmetry",
                    marker_color="purple",
                ),
                row=5,
                col=1,
            )

        fig.update_layout(title_text="Respiration (RSP)", height=1250, width=750)
        for i in range(1, nrow + 1):
            fig.update_xaxes(title_text=x_label, row=i, col=1)

        return fig


# =============================================================================
# Internals
# =============================================================================
def _rsp_plot_phase(rsp_signals, troughs, peaks):
    exhale_signal = pd.Series(np.full(len(rsp_signals), np.nan))
    exhale_signal[troughs] = rsp_signals["RSP_Clean"][troughs].values
    exhale_signal[peaks] = rsp_signals["RSP_Clean"][peaks].values
    exhale_signal = exhale_signal.bfill()

    inhale_signal = pd.Series(np.full(len(rsp_signals), np.nan))
    inhale_signal[troughs] = rsp_signals["RSP_Clean"][troughs].values
    inhale_signal[peaks] = rsp_signals["RSP_Clean"][peaks].values
    inhale_signal = inhale_signal.ffill()

    return exhale_signal, inhale_signal
