# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.collections

def eda_plot(eda_signals, sampling_rate=None):
    """Visualize electrodermal activity (EDA) data.

    Parameters
    ----------
    eda_signals : DataFrame
        DataFrame obtained from `eda_process()`.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> eda_signal = nk.eda_simulate(duration=30, n_scr=5, drift=0.1, noise=0)
    >>> eda_signals, info = nk.eda_process(eda_signal, sampling_rate=1000)
    >>> nk.eda_plot(eda_signals)

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
        x_axis = np.linspace(0, len(eda_signals) / sampling_rate,
                             len(eda_signals))
    else:
        last_ax.set_xlabel("Samples")
        x_axis = np.arange(0, len(eda_signals))

    plt.subplots_adjust(hspace=0.2)

    # Plot cleaned and raw respiration as well as peaks and troughs.
    ax0.set_title("Raw and Cleaned Signal")
    fig.suptitle('Electrodermal Activity (EDA)', fontweight='bold')

    ax0.plot(x_axis, eda_signals["EDA_Raw"], color='#B0BEC5', label='Raw',
             zorder=1)
    ax0.plot(x_axis, eda_signals["EDA_Clean"], color='#9C27B0',
             label='Cleaned', linewidth=1.5, zorder=1)
    ax0.legend(loc='upper right')

    # Plot skin cnoductance response.
    ax1.set_title("Skin Conductance Response (SCR)")
    # Plot Phasic.
    ax1.plot(x_axis, eda_signals["EDA_Phasic"], color='#E91E63', label='Phasic Component', linewidth=1.5, zorder=1)
    ax1.legend(loc='upper right')
    # Mark segments.
    risetime_coord, amplitude_coord, halfr_coord = _eda_plot_segments(eda_signals, ax1, x_axis, onsets, peaks, half_recovery)
    risetime = matplotlib.collections.LineCollection(risetime_coord, colors='#FFA726', linewidths=1, linestyle='dashed')
    ax1.add_collection(risetime)
    amplitude = matplotlib.collections.LineCollection(amplitude_coord, colors='#1976D2', linewidths=1, linestyle='solid')
    ax1.add_collection(amplitude)
    halfr = matplotlib.collections.LineCollection(halfr_coord, colors='#FDD835', linewidths=1, linestyle='dashed')
    ax1.add_collection(halfr)



    # Plot Tonic.
    ax2.set_title("Skin Conductance Level (SCL)")
    ax2.plot(x_axis, eda_signals["EDA_Tonic"], color='#673AB7',
             label='Tonic Component', linewidth=1.5)
    ax2.legend(loc='upper right')
    plt.show()
    return fig




# =============================================================================
# Internals
# =============================================================================
def _eda_plot_segments(eda_signals, ax, x_axis, onsets, peaks, half_recovery):
    # Mark onsets, peaks, and half-recovery.
    scat_onset = ax.scatter(x_axis[onsets], eda_signals["EDA_Phasic"][onsets], color='#FFA726', label="SCR - Onsets", zorder=2)
    scat_peak = ax.scatter(x_axis[peaks], eda_signals["EDA_Phasic"][peaks], color='#1976D2', label="SCR - Peaks", zorder=2)
    scat_halfr = ax.scatter(x_axis[half_recovery], eda_signals["EDA_Phasic"][half_recovery], color='#FDD835', label='SCR - Half-Recovery', zorder=2)
    end_onset = pd.Series(eda_signals["EDA_Phasic"][onsets].values, eda_signals["EDA_Phasic"][peaks].index)
    scat_endonset = ax.scatter(end_onset.index, end_onset.values, alpha=0)

    # Rise time.
    position = [i for i in range(0, len(onsets))]
    risetime_start = scat_onset.get_offsets()
    risetime_end = scat_endonset.get_offsets()
    risetime_coord = [(risetime_start[i], risetime_end[i]) for i in position]

    # SCR Amplitude.
    peak_top = scat_peak.get_offsets()
    amplitude_coord = [(peak_top[i], risetime_end[i]) for i in position]

    # Half recovery.
    halfr_end = scat_halfr.get_offsets()
    halfr_start = [(peak_top[i, 0], halfr_end[i, 1]) for i in position]
    halfr_coord = [(halfr_start[i], halfr_end[i]) for i in position]

    return risetime_coord, amplitude_coord, halfr_coord
