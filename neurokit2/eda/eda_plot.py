# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning


def eda_plot(eda_signals, info=None, static=True):
    """**Visualize electrodermal activity (EDA) data**

    Parameters
    ----------
    eda_signals : DataFrame
        DataFrame obtained from :func:`eda_process()`.
    info : dict
        The information Dict returned by ``eda_process()``. Defaults to ``None``.
    static : bool
        If True, a static plot will be generated with matplotlib.
        If False, an interactive plot will be generated with plotly.
        Defaults to True.

    Returns
    -------
    See :func:`.ecg_plot` for details on how to access the figure, modify the size and save it.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      eda_signal = nk.eda_simulate(duration=30, scr_number=5, drift=0.1, noise=0, sampling_rate=250)
      eda_signals, info = nk.eda_process(eda_signal, sampling_rate=250)

      @savefig p_eda_plot1.png scale=100%
      nk.eda_plot(eda_signals, info)
      @suppress
      plt.close()

    See Also
    --------
    eda_process

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

    # Determine peaks, onsets, and half recovery.
    peaks = np.where(eda_signals["SCR_Peaks"] == 1)[0]
    onsets = np.where(eda_signals["SCR_Onsets"] == 1)[0]
    half_recovery = np.where(eda_signals["SCR_Recovery"] == 1)[0]

    # clean peaks that do not have onsets
    if len(peaks) > len(onsets):
        peaks = peaks[1:]

    # Determine unit of x-axis.
    x_label = "Time (seconds)"
    x_axis = np.linspace(0, len(eda_signals) / info["sampling_rate"], len(eda_signals))

    if static:
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)

        last_ax = fig.get_axes()[-1]
        last_ax.set_xlabel(x_label)

        # Plot cleaned and raw electrodermal activity.
        ax0.set_title("Raw and Cleaned Signal")
        fig.suptitle("Electrodermal Activity (EDA)", fontweight="bold")

        ax0.plot(x_axis, eda_signals["EDA_Raw"], color="#B0BEC5", label="Raw", zorder=1)
        ax0.plot(
            x_axis,
            eda_signals["EDA_Clean"],
            color="#9C27B0",
            label="Cleaned",
            linewidth=1.5,
            zorder=1,
        )
        ax0.legend(loc="upper right")

        # Plot skin conductance response.
        ax1.set_title("Skin Conductance Response (SCR)")

        # Plot Phasic.
        ax1.plot(
            x_axis,
            eda_signals["EDA_Phasic"],
            color="#E91E63",
            label="Phasic Component",
            linewidth=1.5,
            zorder=1,
        )

        # Mark segments.
        risetime_coord, amplitude_coord, halfr_coord = _eda_plot_dashedsegments(
            eda_signals, ax1, x_axis, onsets, peaks, half_recovery
        )

        risetime = matplotlib.collections.LineCollection(
            risetime_coord, colors="#FFA726", linewidths=1, linestyle="dashed"
        )
        ax1.add_collection(risetime)

        amplitude = matplotlib.collections.LineCollection(
            amplitude_coord, colors="#1976D2", linewidths=1, linestyle="solid"
        )
        ax1.add_collection(amplitude)

        halfr = matplotlib.collections.LineCollection(
            halfr_coord, colors="#FDD835", linewidths=1, linestyle="dashed"
        )
        ax1.add_collection(halfr)
        ax1.legend(loc="upper right")

        # Plot Tonic.
        ax2.set_title("Skin Conductance Level (SCL)")
        ax2.plot(
            x_axis,
            eda_signals["EDA_Tonic"],
            color="#673AB7",
            label="Tonic Component",
            linewidth=1.5,
        )
        ax2.legend(loc="upper right")

    else:
        # Create interactive plot with plotly.
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

        except ImportError as e:
            raise ImportError(
                "NeuroKit error: ppg_plot(): the 'plotly'",
                " module is required when 'static' is False.",
                " Please install it first (`pip install plotly`).",
            ) from e

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "Raw and Cleaned Signal",
                "Skin Conductance Response (SCR)",
                "Skin Conductance Level (SCL)",
            ),
        )

        # Plot cleaned and raw electrodermal activity.
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=eda_signals["EDA_Raw"],
                mode="lines",
                name="Raw",
                line=dict(color="#B0BEC5"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=eda_signals["EDA_Clean"],
                mode="lines",
                name="Cleaned",
                line=dict(color="#9C27B0"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Plot skin conductance response.
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=eda_signals["EDA_Phasic"],
                mode="lines",
                name="Phasic Component",
                line=dict(color="#E91E63"),
                showlegend=True,
            ),
            row=2,
            col=1,
        )

        # Mark segments.
        _, _, _ = _eda_plot_dashedsegments(
            eda_signals, fig, x_axis, onsets, peaks, half_recovery, static=static
        )

        # TODO add dashed segments to plotly version

        # Plot skin conductance level.
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=eda_signals["EDA_Tonic"],
                mode="lines",
                name="Tonic Component",
                line=dict(color="#673AB7"),
                showlegend=True,
            ),
            row=3,
            col=1,
        )

        # Add title to entire figure.
        fig.update_layout(title_text="Electrodermal Activity (EDA)", title_x=0.5)

        return fig


# =============================================================================
# Internals
# =============================================================================
def _eda_plot_dashedsegments(
    eda_signals, ax, x_axis, onsets, peaks, half_recovery, static=True
):
    # Mark onsets, peaks, and half-recovery.
    onset_x_values = x_axis[onsets]
    onset_y_values = eda_signals["EDA_Phasic"][onsets].values
    peak_x_values = x_axis[peaks]
    peak_y_values = eda_signals["EDA_Phasic"][peaks].values
    halfr_x_values = x_axis[half_recovery]
    halfr_y_values = eda_signals["EDA_Phasic"][half_recovery].values

    end_onset = pd.Series(
        eda_signals["EDA_Phasic"][onsets].values, eda_signals["EDA_Phasic"][peaks].index
    )

    risetime_coord = []
    amplitude_coord = []
    halfr_coord = []

    for i in range(len(onsets)):
        # Rise time.
        start = (onset_x_values[i], onset_y_values[i])
        end = (peak_x_values[i], onset_y_values[i])
        risetime_coord.append((start, end))

    for i in range(len(peaks)):
        # SCR Amplitude.
        start = (peak_x_values[i], onset_y_values[i])
        end = (peak_x_values[i], peak_y_values[i])
        amplitude_coord.append((start, end))

    for i in range(len(half_recovery)):
        # Half recovery.
        end = (halfr_x_values[i], halfr_y_values[i])
        peak_x_idx = np.where(peak_x_values < halfr_x_values[i])[0][-1]
        start = (peak_x_values[peak_x_idx], halfr_y_values[i])
        halfr_coord.append((start, end))

    if static:
        # Plot with matplotlib.
        # Mark onsets, peaks, and half-recovery.
        ax.scatter(
            x_axis[onsets],
            eda_signals["EDA_Phasic"][onsets],
            color="#FFA726",
            label="SCR - Onsets",
            zorder=2,
        )
        ax.scatter(
            x_axis[peaks],
            eda_signals["EDA_Phasic"][peaks],
            color="#1976D2",
            label="SCR - Peaks",
            zorder=2,
        )
        ax.scatter(
            x_axis[half_recovery],
            eda_signals["EDA_Phasic"][half_recovery],
            color="#FDD835",
            label="SCR - Half recovery",
            zorder=2,
        )

        ax.scatter(x_axis[end_onset.index], end_onset.values, alpha=0)
    else:
        # Create interactive plot with plotly.
        try:
            import plotly.graph_objects as go

        except ImportError as e:
            raise ImportError(
                "NeuroKit error: ppg_plot(): the 'plotly'",
                " module is required when 'static' is False.",
                " Please install it first (`pip install plotly`).",
            ) from e
        # Plot with plotly.
        # Mark onsets, peaks, and half-recovery.
        ax.add_trace(
            go.Scatter(
                x=x_axis[onsets],
                y=eda_signals["EDA_Phasic"][onsets],
                mode="markers",
                name="SCR - Onsets",
                marker=dict(color="#FFA726"),
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        ax.add_trace(
            go.Scatter(
                x=x_axis[peaks],
                y=eda_signals["EDA_Phasic"][peaks],
                mode="markers",
                name="SCR - Peaks",
                marker=dict(color="#1976D2"),
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        ax.add_trace(
            go.Scatter(
                x=x_axis[half_recovery],
                y=eda_signals["EDA_Phasic"][half_recovery],
                mode="markers",
                name="SCR - Half recovery",
                marker=dict(color="#FDD835"),
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        ax.add_trace(
            go.Scatter(
                x=x_axis[end_onset.index],
                y=end_onset.values,
                mode="markers",
                marker=dict(color="#FDD835", opacity=0),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    return risetime_coord, amplitude_coord, halfr_coord
