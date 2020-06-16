# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from .density import density
from .rescale import rescale


def summary_plot(x, **kwargs):
    """Descriptive plot.

    Visualize a distribution with density, histogram, boxplot and rugs plots all at once.

    Examples
    --------
    >>> import neurokit2 as nk
    >>> import numpy as np
    >>>
    >>> x = np.random.normal(size=100)
    >>> fig = nk.summary_plot(x)
    >>> fig #doctest: +SKIP

    """

    if "ax" in kwargs:
        fig = None
        ax = kwargs.get("ax")
        kwargs.pop("ax")
    else:
        fig, ax = plt.subplots()

    # Histogram
    counts, bins = np.histogram(x, **kwargs)
    ax.hist(bins[:-1], bins, weights=counts, color="#2196F3", edgecolor="white", zorder=1, **kwargs)

    # Density
    x_axis, y_axis = density(x, **kwargs)
    y_axis = rescale(y_axis, to=[0, np.max(counts)])
    ax.plot(x_axis, y_axis, color="#E91E63", linewidth=1.5, zorder=2, **kwargs)

    # Points
    y_axis = np.full(len(x), 0.1)
    ax.scatter(x, y_axis, c="black", alpha=0.5, marker="|", zorder=3, **kwargs)

    # Boxplot
    ax.boxplot(
        x,
        vert=False,
        positions=[np.max(counts) / 10],
        widths=np.max(counts) / 10,
        manage_ticks=False,
        boxprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        zorder=4,
        **kwargs
    )
    return fig
