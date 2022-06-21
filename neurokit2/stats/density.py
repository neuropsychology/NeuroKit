# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.stats

from .density_bandwidth import density_bandwidth


def density(x, desired_length=100, bandwidth="Scott", show=False, **kwargs):
    """Density estimation.

    Computes kernel density estimates.

    Parameters
    -----------
    x : Union[list, np.array, pd.Series]
        A vector of values.
    desired_length : int
        The amount of values in the returned density estimation.
    bandwidth : float
        Passed to the ``method`` argument from the :func:`.density_bandwidth` function.
    show : bool
        Display the density plot.
    **kwargs
        Additional arguments to be passed to :func:`.density_bandwidth`.

    Returns
    -------
    x
        The x axis of the density estimation.
    y
        The y axis of the density estimation.

    See Also
    --------
    density_bandwidth

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.ecg_simulate(duration=20)

      @savefig p_density1.png scale=100%
      x, y = nk.density(signal, bandwidth=0.5, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      # Bandwidth comparison
      _, y2 = nk.density(signal, bandwidth=1)
      _, y3 = nk.density(signal, bandwidth=2)
      _, y4 = nk.density(signal, bandwidth="scott")
      _, y5 = nk.density(signal, bandwidth="silverman")
      _, y6 = nk.density(signal, bandwidth="kernsmooth")

      @savefig p_density2.png scale=100%
      nk.signal_plot([y, y2, y3, y4, y5, y6],
                     labels=["0.5", "1", "2", "Scott", "Silverman", "KernSmooth"])
      @suppress
      plt.close()

    """
    if "method" in kwargs:
        kwargs.pop("method")
    bw = density_bandwidth(x, method=bandwidth, **kwargs)
    density_function = scipy.stats.gaussian_kde(x, bw_method=bw)

    x = np.linspace(np.nanmin(x), np.nanmax(x), num=desired_length)
    y = density_function(x)

    if show is True:
        pd.DataFrame({"x": x, "y": y}).plot(x="x")

    return x, y
