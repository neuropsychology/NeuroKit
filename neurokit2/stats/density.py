# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.stats


def density(x, desired_length=100, bandwith=1, show=False):
    """Density estimation.

    Computes kernel density estimates.

    Parameters
    -----------
    x : Union[list, np.array, pd.Series]
        A vector of values.
    desired_length : int
        The amount of values in the returned density estimation.
    bandwith : float
        The bandwith of the kernel. The smaller the values, the smoother the estimation.
    show : bool
        Display the density plot.

    Returns
    -------
    x, y
        The x axis of the density estimation.
    y
        The y axis of the density estimation.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.ecg_simulate(duration=20)
    >>> x, y = nk.density(signal, bandwith=0.5, show=True)
    >>>
    >>> # Bandwidth comparison
    >>> x, y1 = nk.density(signal, bandwith=0.5)
    >>> x, y2 = nk.density(signal, bandwith=1)
    >>> x, y3 = nk.density(signal, bandwith=2)
    >>> pd.DataFrame({"x": x, "y1": y1, "y2": y2, "y3": y3}).plot(x="x") #doctest: +SKIP

    """
    density_function = scipy.stats.gaussian_kde(x, bw_method="scott")
    density_function.set_bandwidth(bw_method=density_function.factor / bandwith)

    x = np.linspace(np.min(x), np.max(x), num=desired_length)
    y = density_function(x)

    if show is True:
        pd.DataFrame({"x": x, "y": y}).plot(x="x")

    return x, y
