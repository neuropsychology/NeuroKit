# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal

from ..events.events_plot import events_plot


def find_plateau(values, show=True):

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
    largest = np.argsort(slope_change)[:int(0.1 * len(slope_change))]  # get top 10%
    optimal = [i for i in largest if i in indices]

    # find indices above certain threshold
    # threshold = np.where(values >= np.percentile(values, 80))[0]
    # optimal = np.intersect1d(indices, threshold)

    if len(optimal) >= 1:
        plateau = np.where(values == np.max(values[optimal]))[0][0]
        if show:
            events_plot([plateau], values)
    else:
        plateau = None

    return plateau
