# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from ..signal.signal_rate import _signal_rate_plot
from .ppg_peaks import _ppg_peaks_plot
from .ppg_segment import ppg_segment


def ppg_plot(ppg_signals, info=None, static=True):
    """**Visualize photoplethysmogram (PPG) data**

    Visualize the PPG signal processing.

    Parameters
    ----------
    ppg_signals : DataFrame
        DataFrame obtained from :func:`.ppg_process`.
    info : dict
        The information Dict returned by ``ppg_process()``. Defaults to ``None``.
    static : bool
        If True, a static plot will be generated with matplotlib.
        If False, an interactive plot will be generated with plotly.
        Defaults to True.

    Returns
    -------
    See :func:`.ecg_plot` for details on how to access the figure, modify the size and save it.

    See Also
    --------
    ppg_process

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate data
      ppg = nk.ppg_simulate(duration=10, sampling_rate=100, heart_rate=70)
      # Process signal
      signals, info = nk.ppg_process(ppg, sampling_rate=100)

      # Plot
      @savefig p_ppg_plot1.png scale=100%
      nk.ppg_plot(signals, info)
      @suppress
      plt.close()

    """

    # Sanity-check input.
    if not isinstance(ppg_signals, pd.DataFrame):
        raise ValueError(
            "NeuroKit error: The `ppg_signals` argument must"
            " be the DataFrame returned by `ppg_process()`."
        )

    # Extract Peaks.
    if info is None:
        warn(
            "'info' dict not provided. Some information might be missing."
            + " Sampling rate will be set to 1000 Hz.",
            category=NeuroKitWarning,
        )
        info = {"sampling_rate": 1000}

    # Extract Peaks (take those from df as it might have been cropped)
    if "PPG_Peaks" in ppg_signals.columns:
        info["PPG_Peaks"] = np.where(ppg_signals["PPG_Peaks"] == 1)[0]

    if static:
        # Prepare figure
        gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[2 / 3, 1 / 3])
        fig = plt.figure(constrained_layout=False)

        ax0 = fig.add_subplot(gs[0, :-1])
        ax1 = fig.add_subplot(gs[1, :-1], sharex=ax0)
        ax2 = fig.add_subplot(gs[:, -1])

        fig.suptitle("Photoplethysmogram (PPG)", fontweight="bold")

        # Plot cleaned and raw PPG
        ax0 = _ppg_peaks_plot(
            ppg_signals["PPG_Clean"].values,
            info=info,
            sampling_rate=info["sampling_rate"],
            raw=ppg_signals["PPG_Raw"].values,
            quality=ppg_signals["PPG_Quality"].values,
            ax=ax0,
        )

        # Plot Heart Rate
        ax1 = _signal_rate_plot(
            ppg_signals["PPG_Rate"].values,
            info["PPG_Peaks"],
            sampling_rate=info["sampling_rate"],
            title="Heart Rate",
            ytitle="Beats per minute (bpm)",
            color="#FB661C",
            color_mean="#FBB41C",
            color_points="#FF9800",
            ax=ax1,
        )

        # Plot individual heart beats
        ax2 = ppg_segment(
            ppg_signals["PPG_Clean"].values,
            info["PPG_Peaks"],
            info["sampling_rate"],
            show="return",
            ax=ax2,
        )

    else:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

        except ImportError as e:
            raise ImportError(
                "NeuroKit error: ppg_plot(): the 'plotly'",
                " module is required when 'static' is False.",
                " Please install it first (`pip install plotly`).",
            ) from e

        # X-axis
        x_axis = np.linspace(
            0, len(ppg_signals) / info["sampling_rate"], len(ppg_signals)
        )

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Raw and Cleaned Signal", "Rate"),
        )

        # Plot cleaned and raw PPG
        fig.add_trace(
            go.Scatter(x=x_axis, y=ppg_signals["PPG_Raw"], name="Raw"), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=ppg_signals["PPG_Clean"],
                name="Cleaned",
                marker_color="#FB1CF0",
            ),
            row=1,
            col=1,
        )

        # Plot peaks
        fig.add_trace(
            go.Scatter(
                x=x_axis[info["PPG_Peaks"]],
                y=ppg_signals["PPG_Clean"][info["PPG_Peaks"]],
                name="Peaks",
                mode="markers",
                marker_color="#D60574",
            ),
            row=1,
            col=1,
        )

        # Rate
        ppg_rate_mean = ppg_signals["PPG_Rate"].mean()
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=ppg_signals["PPG_Rate"],
                name="Rate",
                mode="lines",
                marker_color="#FB661C",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=ppg_rate_mean,
            line_dash="dash",
            line_color="#FBB41C",
            name="Mean",
            row=2,
            col=1,
        )
        fig.update_layout(title_text="Photoplethysmogram (PPG)", height=500, width=750)
        if info["sampling_rate"] is not None:
            fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
            fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        elif info["sampling_rate"] is None:
            fig.update_xaxes(title_text="Samples", row=1, col=1)
            fig.update_xaxes(title_text="Samples", row=2, col=1)
        return fig
