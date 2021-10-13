# -*- coding: utf-8 -*-
import numpy as np

from ..events.events_plot import events_plot


def find_plateau(values, show=True):

    # find indices in increasing segments
    increasing_segments = np.where(np.diff(values) > 0)[0]

    # find indices where positive gradients are becoming less positive
    slope_change = np.diff(np.diff(values))
    gradients = np.where(slope_change < 0)[0]
    indices = np.intersect1d(increasing_segments, gradients)

    # find greatest change in slopes
    largest = np.argsort(slope_change)[:5] + 2 # get 5 largest elements
    optimal = [i for i in largest if i in indices][0]

    if show:
        events_plot([optimal], values)

    return optimal
