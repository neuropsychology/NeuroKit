# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..epochs import epochs_create, epochs_to_array, epochs_to_df
from ..misc import NeuroKitWarning
from ..stats import standardize


def eog_plot(eog_signals, info=None):
    """**Visualize EOG data**

    Parameters
    ----------
    eog_signals : DataFrame
        DataFrame obtained from :func:`.eog_process`.
    info : dict
        The information Dict returned by ``eog_process()``. Defaults to ``None``.

    See Also
    --------
    eog_process

    Returns
    -------
    Though the function returns nothing, the figure can be retrieved and saved as follows:

    .. code-block:: console

        # To be run after eog_plot()
        fig = plt.gcf()
        fig.savefig("myfig.png")

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate data
      eog_signal = nk.data('eog_100hz')

      # Process signal
      eog_signals, info = nk.eog_process(eog_signal, sampling_rate=100)

      # Plot
      @savefig p_eog_plot.png scale=100%
      nk.eog_plot(eog_signals, info)
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

    # Sanity-check input.
    if not isinstance(eog_signals, pd.DataFrame):
        raise ValueError(
            "NeuroKit error: eog_plot(): The `eog_signals` argument must"
            " be the DataFrame returned by `eog_process()`."
        )

    # Prepare figure
    x_axis = np.linspace(
        0, eog_signals.shape[0] / info["sampling_rate"], eog_signals.shape[0]
    )
    gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[1 - 1 / np.pi, 1 / np.pi])
    fig = plt.figure(constrained_layout=False)
    ax0 = fig.add_subplot(gs[0, :-1])
    ax1 = fig.add_subplot(gs[1, :-1])
    ax2 = fig.add_subplot(gs[:, -1])
    ax0.set_xlabel("Time (seconds)")
    ax1.set_xlabel("Time (seconds)")
    ax2.set_xlabel("Time (seconds)")

    fig.suptitle("Electrooculography (EOG)", fontweight="bold")
    plt.tight_layout(h_pad=0.3, w_pad=0.2)

    # Plot cleaned and raw EOG
    ax0.set_title("Raw and Cleaned Signal")
    ax0.plot(x_axis, eog_signals["EOG_Raw"], color="#B0BEC5", label="Raw", zorder=1)
    ax0.plot(
        x_axis,
        eog_signals["EOG_Clean"],
        color="#49A4FD",
        label="Cleaned",
        zorder=1,
        linewidth=1.5,
    )
    ax0.set_ylabel("Amplitude (mV)")

    # Plot blinks
    blinks = np.where(eog_signals["EOG_Blinks"] == 1)[0]
    ax0.scatter(
        x_axis[blinks],
        eog_signals["EOG_Clean"][blinks],
        color="#0146D7",
        label="Blinks",
        zorder=2,
    )
    ax0.legend(loc="upper right")

    # Rate
    ax1.set_title("Blink Rate")
    ax1.set_ylabel("Blinks per minute")
    blink_rate_mean = eog_signals["EOG_Rate"].mean()
    ax1.plot(
        x_axis, eog_signals["EOG_Rate"], color="#9C5AFF", label="Rate", linewidth=1.5
    )
    ax1.axhline(y=blink_rate_mean, label="Mean", linestyle="--", color="#CEAFFF")
    ax1.legend(loc="upper right")

    # Plot individual blinks
    ax2.set_title("Individual Blinks")

    # Create epochs
    events = epochs_create(
        eog_signals["EOG_Clean"],
        info["EOG_Blinks"],
        sampling_rate=info["sampling_rate"],
        epochs_start=-0.3,
        epochs_end=0.7,
    )
    events_array = epochs_to_array(events)  # Convert to 2D array
    events_array = standardize(
        events_array
    )  # Rescale so that all the blinks are on the same scale

    blinks_df = epochs_to_df(events)
    blinks_wide = blinks_df.pivot(index="Time", columns="Label", values="Signal")
    blinks_wide = standardize(blinks_wide)

    cmap = iter(plt.cm.RdBu(np.linspace(0, 1, num=len(events))))
    for x, color in zip(blinks_wide, cmap):
        ax2.plot(blinks_wide[x], color=color, linewidth=0.4, zorder=1)

    # Plot with their median (used here as a robust average)
    ax2.plot(
        np.array(blinks_wide.index),
        np.median(events_array, axis=1),
        linewidth=2,
        linestyle="--",
        color="black",
        label="Median",
    )
    ax2.legend(loc="upper right")
