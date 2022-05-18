# -*- coding: utf-8 -*-
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from ..events.events_plot import events_plot


def find_plateau(values, show=True):
    """**Find the point of plateau in an array of values**

    Parameters
    ----------
    values : ndarray
        An array of values.
    show : bool
        If ``True``, will return the plot visualizing the trajectory and point of plateau.

    Returns
    ----------
    plateau : int
        The index of the plateau.

    Example
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Generate values manually
      x = np.linspace(1, 5, 50)
      y = 2.5 * np.log(1.3 * x) + 0.5
      y = y + 0.05 * np.random.normal(size=len(x))

      # Find plateau
      @savefig p_find_plateau1.png scale=100%
      plateau = nk.find_plateau(y, show=True)
      @suppress
      plt.close()
      plateau


    """

    # find indices in increasing segments
    increasing_segments = np.where(np.diff(values) > 0)[0]

    # get indices where positive gradients are becoming less positive
    slope_change = np.diff(np.diff(values))
    gradients = np.where(slope_change < 0)[0]
    indices = np.intersect1d(increasing_segments, gradients)

    # exclude inverse peaks
    peaks = scipy.signal.find_peaks(-1 * values)[0]
    if len(peaks) > 0:
        indices = [i for i in indices if i not in peaks]

    # find greatest change in slopes amongst filtered indices
    largest = np.argsort(slope_change)[: int(0.1 * len(slope_change))]  # get top 10%
    optimal = [i for i in largest if i in indices]

    if len(optimal) >= 1:
        plateau = np.where(values == np.max(values[optimal]))[0][0]
        if show:
            events_plot([plateau], values)
            # _find_plateau_plot(values, increasing_segments, indices, optimal, plateau)
    else:
        plateau = None

    return plateau


def _find_plateau_plot(values, increasing_segments, indices, optimal, plateau):
    """For visualization of the steps involved in `find_plateau()`"""

    # Prepare fig
    fig = plt.figure(constrained_layout=False)
    spec = gs.GridSpec(ncols=2, nrows=2, height_ratios=[1, 1], width_ratios=[1, 1])
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[1, 0])
    ax4 = fig.add_subplot(spec[1, 1])

    # Plot
    ax1.plot(values)
    ax1.set_title("Points of increasing segments")
    for i in increasing_segments:
        ax1.axvline(x=i, color="red", linestyle="--")

    ax2.plot(values)
    ax2.set_title("Points of decelerating positive gradients")
    for i in indices:
        ax2.axvline(x=i, color="blue", linestyle="--")

    ax3.plot(values)
    ax3.set_title("Points of greatest slope changes")
    for i in optimal:
        ax3.axvline(x=i, color="purple", linestyle="--")

    ax4.plot(values)
    ax4.set_title("Optimal Point")
    ax4.axvline(x=plateau, color="orange", linestyle="--")

    return fig
