# -*- coding: utf-8 -*-
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import find_closest


def eda_plot(eda_signals, sampling_rate=None):
    """Visualize electrodermal activity (EDA) data.

    Parameters
    ----------
    eda_signals : DataFrame
        DataFrame obtained from `eda_process()`.
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second). Defaults to None.

    Returns
    -------
    fig
        Figure representing a plot of the processed EDA signals.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> eda_signal = nk.eda_simulate(duration=30, scr_number=5, drift=0.1, noise=0, sampling_rate=250)
    >>> eda_signals, info = nk.eda_process(eda_signal, sampling_rate=250)
    >>> fig = nk.eda_plot(eda_signals)
    >>> fig #doctest: +SKIP

    See Also
    --------
    eda_process

    """
    # Determine peaks, onsets, and half recovery.
    peaks = np.where(eda_signals["SCR_Peaks"] == 1)[0]
    onsets = np.where(eda_signals["SCR_Onsets"] == 1)[0]
    half_recovery = np.where(eda_signals["SCR_Recovery"] == 1)[0]

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)

    # Determine unit of x-axis.
    last_ax = fig.get_axes()[-1]
    if sampling_rate is not None:
        last_ax.set_xlabel("Seconds")
        x_axis = np.linspace(0, len(eda_signals) / sampling_rate, len(eda_signals))
    else:
        last_ax.set_xlabel("Samples")
        x_axis = np.arange(0, len(eda_signals))

    plt.subplots_adjust(hspace=0.2)

    # Plot cleaned and raw respiration as well as peaks and troughs.
    ax0.set_title("Raw and Cleaned Signal")
    fig.suptitle("Electrodermal Activity (EDA)", fontweight="bold")

    ax0.plot(x_axis, eda_signals["EDA_Raw"], color="#B0BEC5", label="Raw", zorder=1)
    ax0.plot(x_axis, eda_signals["EDA_Clean"], color="#9C27B0", label="Cleaned", linewidth=1.5, zorder=1)
    ax0.legend(loc="upper right")

    # Plot skin cnoductance response.
    ax1.set_title("Skin Conductance Response (SCR)")

    # Plot Phasic.
    ax1.plot(x_axis, eda_signals["EDA_Phasic"], color="#E91E63", label="Phasic Component", linewidth=1.5, zorder=1)

    # Mark segments.
    risetime_coord, amplitude_coord, halfr_coord = _eda_plot_dashedsegments(
        eda_signals, ax1, x_axis, onsets, peaks, half_recovery
    )

    risetime = matplotlib.collections.LineCollection(risetime_coord, colors="#FFA726", linewidths=1, linestyle="dashed")
    ax1.add_collection(risetime)

    amplitude = matplotlib.collections.LineCollection(
        amplitude_coord, colors="#1976D2", linewidths=1, linestyle="solid"
    )
    ax1.add_collection(amplitude)

    halfr = matplotlib.collections.LineCollection(halfr_coord, colors="#FDD835", linewidths=1, linestyle="dashed")
    ax1.add_collection(halfr)
    ax1.legend(loc="upper right")

    # Plot Tonic.
    ax2.set_title("Skin Conductance Level (SCL)")
    ax2.plot(x_axis, eda_signals["EDA_Tonic"], color="#673AB7", label="Tonic Component", linewidth=1.5)
    ax2.legend(loc="upper right")
    plt.show()
    return fig


# =============================================================================
# Internals
# =============================================================================
def _eda_plot_dashedsegments(eda_signals, ax, x_axis, onsets, peaks, half_recovery):
    # Mark onsets, peaks, and half-recovery.
    scat_onset = ax.scatter(
        x_axis[onsets], eda_signals["EDA_Phasic"][onsets], color="#FFA726", label="SCR - Onsets", zorder=2
    )
    scat_peak = ax.scatter(
        x_axis[peaks], eda_signals["EDA_Phasic"][peaks], color="#1976D2", label="SCR - Peaks", zorder=2
    )
    scat_halfr = ax.scatter(
        x_axis[half_recovery],
        eda_signals["EDA_Phasic"][half_recovery],
        color="#FDD835",
        label="SCR - Half recovery",
        zorder=2,
    )
    end_onset = pd.Series(eda_signals["EDA_Phasic"][onsets].values, eda_signals["EDA_Phasic"][peaks].index)
    scat_endonset = ax.scatter(x_axis[end_onset.index], end_onset.values, alpha=0)

    # Rise time.
    risetime_start = scat_onset.get_offsets()
    risetime_end = scat_endonset.get_offsets()
    risetime_coord = [(risetime_start[i], risetime_end[i]) for i in range(0, len(onsets))]

    # SCR Amplitude.
    peak_top = scat_peak.get_offsets()
    amplitude_coord = [(peak_top[i], risetime_end[i]) for i in range(0, len(onsets))]

    # Half recovery.
    peak_x_values = peak_top.data[:, 0]
    recovery_x_values = x_axis[half_recovery]

    peak_list = []
    for i, index in enumerate(half_recovery):
        value = find_closest(recovery_x_values[i], peak_x_values, direction="smaller", strictly=False)
        peak_list.append(value)

    peak_index = []
    for i in np.array(peak_list):
        index = np.where(i == peak_x_values)[0][0]
        peak_index.append(index)

    halfr_index = list(range(0, len(half_recovery)))
    halfr_end = scat_halfr.get_offsets()
    halfr_start = [(peak_top[i, 0], halfr_end[x, 1]) for i, x in zip(peak_index, halfr_index)]
    halfr_coord = [(halfr_start[i], halfr_end[i]) for i in halfr_index]

    return risetime_coord, amplitude_coord, halfr_coord
