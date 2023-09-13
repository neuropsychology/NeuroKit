# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning


def emg_plot(emg_signals, info=None, static=True):
    """**EMG Graph**

    Visualize electromyography (EMG) data.

    Parameters
    ----------
    emg_signals : DataFrame
        DataFrame obtained from ``emg_process()``.
    info : dict
        The information Dict returned by ``emg_process()``. Defaults to ``None``.
    static : bool
        If True, a static plot will be generated with matplotlib.
        If False, an interactive plot will be generated with plotly.
        Defaults to True.

    See Also
    --------
    emg_process

    Returns
    -------
    See :func:`.ecg_plot` for details on how to access the figure, modify the size and save it.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate data
      emg = nk.emg_simulate(duration=10, sampling_rate=1000, burst_number=3)

      # Process signal
      emg_signals, info = nk.emg_process(emg, sampling_rate=1000)

      # Plot
      @savefig p_emg_plot.png scale=100%
      nk.emg_plot(emg_signals, info)
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

    # Mark onsets, offsets, activity
    onsets = np.where(emg_signals["EMG_Onsets"] == 1)[0]
    offsets = np.where(emg_signals["EMG_Offsets"] == 1)[0]

    # Sanity-check input.
    if not isinstance(emg_signals, pd.DataFrame):
        raise ValueError(
            "NeuroKit error: The `emg_signals` argument must"
            " be the DataFrame returned by `emg_process()`."
        )

    # Determine what to display on the x-axis, mark activity.
    x_axis = np.linspace(
        0, emg_signals.shape[0] / info["sampling_rate"], emg_signals.shape[0]
    )

    if static is True:
        _emg_plot_static(emg_signals, x_axis, onsets, offsets, info["sampling_rate"])
    else:
        return _emg_plot_interactive(
            emg_signals, x_axis, onsets, offsets, info["sampling_rate"]
        )


# =============================================================================
# Internals
# =============================================================================
def _emg_plot_activity(emg_signals, onsets, offsets):
    activity_signal = pd.Series(np.full(len(emg_signals), np.nan))
    activity_signal[onsets] = emg_signals["EMG_Amplitude"][onsets].values
    activity_signal[offsets] = emg_signals["EMG_Amplitude"][offsets].values
    activity_signal = activity_signal.bfill()

    if np.any(activity_signal.isna()):
        index = np.min(np.where(activity_signal.isna())) - 1
        value_to_fill = activity_signal[index]
        activity_signal = activity_signal.fillna(value_to_fill)

    return activity_signal


def _emg_plot_static(emg_signals, x_axis, onsets, offsets, sampling_rate):
    # Prepare figure.
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    if sampling_rate is not None:
        ax1.set_xlabel("Time (seconds)")
    elif sampling_rate is None:
        ax1.set_xlabel("Samples")

    fig.suptitle("Electromyography (EMG)", fontweight="bold")
    plt.tight_layout(h_pad=0.2)

    # Plot cleaned and raw EMG.
    ax0.set_title("Raw and Cleaned Signal")
    ax0.plot(x_axis, emg_signals["EMG_Raw"], color="#B0BEC5", label="Raw", zorder=1)
    ax0.plot(
        x_axis,
        emg_signals["EMG_Clean"],
        color="#FFC107",
        label="Cleaned",
        zorder=1,
        linewidth=1.5,
    )
    ax0.legend(loc="upper right")

    # Plot Amplitude.
    ax1.set_title("Muscle Activation")
    ax1.plot(
        x_axis,
        emg_signals["EMG_Amplitude"],
        color="#FF9800",
        label="Amplitude",
        linewidth=1.5,
    )

    # Shade activity regions.
    activity_signal = _emg_plot_activity(emg_signals, onsets, offsets)
    ax1.fill_between(
        x_axis,
        emg_signals["EMG_Amplitude"],
        activity_signal,
        where=emg_signals["EMG_Amplitude"] > activity_signal,
        color="#f7c568",
        alpha=0.5,
        label=None,
    )

    # Mark onsets and offsets.
    ax1.scatter(
        x_axis[onsets],
        emg_signals["EMG_Amplitude"][onsets],
        color="#f03e65",
        label=None,
        zorder=3,
    )
    ax1.scatter(
        x_axis[offsets],
        emg_signals["EMG_Amplitude"][offsets],
        color="#f03e65",
        label=None,
        zorder=3,
    )

    if sampling_rate is not None:
        onsets = onsets / sampling_rate
        offsets = offsets / sampling_rate

    for i, j in zip(list(onsets), list(offsets)):
        ax1.axvline(i, color="#4a4a4a", linestyle="--", label=None, zorder=2)
        ax1.axvline(j, color="#4a4a4a", linestyle="--", label=None, zorder=2)
    ax1.legend(loc="upper right")


def _emg_plot_interactive(emg_signals, x_axis, onsets, offsets, sampling_rate):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError(
            "NeuroKit error: emg_plot(): the 'plotly' "
            "module is required for this feature."
            "Please install it first (`pip install plotly`)."
        )

    # Prepare figure.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.update_layout(title="Electromyography (EMG)", font=dict(size=18), height=600)

    # Plot cleaned and raw EMG.
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=emg_signals["EMG_Raw"],
            mode="lines",
            name="Raw",
            line=dict(color="#B0BEC5"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=emg_signals["EMG_Clean"],
            mode="lines",
            name="Cleaned",
            line=dict(color="#FFC107"),
        ),
        row=1,
        col=1,
    )

    # Plot Amplitude.
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=emg_signals["EMG_Amplitude"],
            mode="lines",
            name="Amplitude",
            line=dict(color="#FF9800"),
        ),
        row=2,
        col=1,
    )

    # Mark onsets and offsets.
    fig.add_trace(
        go.Scatter(
            x=x_axis[onsets],
            y=emg_signals["EMG_Amplitude"][onsets],
            mode="markers",
            name="Onsets",
            marker=dict(color="#f03e65", size=10),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis[offsets],
            y=emg_signals["EMG_Amplitude"][offsets],
            mode="markers",
            name="Offsets",
            marker=dict(color="#f03e65", size=10),
        ),
        row=2,
        col=1,
    )

    if sampling_rate is not None:
        onsets = onsets / sampling_rate
        offsets = offsets / sampling_rate
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    elif sampling_rate is None:
        fig.update_xaxes(title_text="Samples", row=2, col=1)

    for i, j in zip(list(onsets), list(offsets)):
        fig.add_shape(
            type="line",
            x0=i,
            y0=0,
            x1=i,
            y1=1,
            line=dict(color="#4a4a4a", width=2, dash="dash"),
            row=2,
            col=1,
        )
        fig.add_shape(
            type="line",
            x0=j,
            y0=0,
            x1=j,
            y1=1,
            line=dict(color="#4a4a4a", width=2, dash="dash"),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    return fig
