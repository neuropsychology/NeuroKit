# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


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
    >>> eda = nk.eda_simulate(duration=30, n_scr=5, drift=0.1, noise=0)
    >>> signals, info = nk.eda_process(eda, sampling_rate=1000)
    >>> nk.eda_plot(signals)

    See Also
    --------
    eda_process
    """
    peaks = np.where(eda_signals["SCR_Peaks"] == 1)[0]

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
    ax0.set_title("Raw and Cleaned EDA")
    fig.suptitle('Electrodermal Activity (EDA)', fontweight='bold')

    ax0.plot(x_axis, eda_signals["EDA_Raw"], color='#B0BEC5', label='Raw',
             zorder=1)
    ax0.plot(x_axis, eda_signals["EDA_Clean"], color='#9C27B0',
             label='Cleaned', zorder=1)
    ax0.legend(loc='upper right')





    # Plot Phasic.
    ax1.set_title("Phasic Component")
    ax1.plot(x_axis, eda_signals["EDA_Phasic"], color='#E91E63', label='Phasic')

    # Add peaks
    ax1.scatter(x_axis[peaks], eda_signals["EDA_Phasic"][peaks], color='#FF9800',
                label="Skin Conductance Responses (SCRs)", zorder=2)
    ax1.legend(loc='upper right')

    # Plot Tonic.
    ax2.set_title("Tonic Component")
    ax2.plot(x_axis, eda_signals["EDA_Tonic"], color='#673AB7',
             label='Tonic')
    ax2.legend(loc='upper right')

    plt.show()
    return fig
