from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from .entropy_shannon import entropy_shannon


def entropy_grid(signal, delay=1, k=3, show=False, **kwargs):
    """**Grid Entropy (GridEn)**

    Grid Entropy (GridEn or GDEn) is defined as a gridded descriptor of a :func:`Poincaré plot <.hrv_nonlinear>`,
    which is a two-dimensional phase space diagram of a time series that plots the present sample
    of a time series with respect to their delayed values. The signal is normalized to the range
    [0, 1] (min-max scaling), and the square of the plot is divided into :math:`k \\times k` grids
    of equal size. The :func:`Shannon entropy <entropy_shannon>` is computed from the probability
    distribution of the number of points in each grid (the grids without points do not contribute),
    so that GridEn lies between 0 and :math:`\\log(k^2)`.

    Yan et al. (2019) define two novel measures, namely **GridEn** and **Gridded Distribution Rate
    (GDR)**, the latter being the proportion of grids containing points.


    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    k : int
        The number of sections into which each axis of the Poincaré plot is divided (giving
        :math:`k \\times k` grids). It is a coarse graining parameter that defines how fine the
        grid is.
    show : bool
        Plot the Poincaré plot and the number of points in each grid.
    **kwargs : optional
        Other keyword arguments, such as the logarithmic ``base`` to use for
        :func:`entropy_shannon` (2 by default, giving bits).

    Returns
    -------
    griden : float
        Grid Entropy of the signal. It is NaN (with a warning) for signals with missing values or
        that are constant.
    info : dict
        A dictionary containing additional information regarding the parameters used (``k`` and
        ``Delay``), and the Gridded Distribution Rate (``GDR``).

    See Also
    --------
    entropy_shannon, .hrv_nonlinear, entropy_phase

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal
      signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)

      # Compute Grid Entropy
      @savefig p_entropy_grid1.png scale=100%
      griden, info = nk.entropy_grid(signal, k=3, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      griden

      @savefig p_entropy_grid2.png scale=100%
      griden, info = nk.entropy_grid(signal, k=10, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      info["GDR"]

    References
    ----------
    * Yan, C., Li, P., Liu, C., Wang, X., Yin, C., & Yao, L. (2019). Novel gridded descriptors of
      Poincaré plot for analyzing heartbeat interval time-series. Computers in Biology and
      Medicine, 109, 280-289.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    signal = np.asarray(signal, dtype=float)
    if k < 1 or int(k) != k:
        raise ValueError("The number of sections `k` of Grid Entropy should be a positive integer.")
    if delay < 1 or int(delay) != delay:
        raise ValueError("The `delay` of Grid Entropy should be a positive integer.")
    if len(signal) <= delay:
        raise ValueError("The signal should be longer than `delay` to compute Grid Entropy.")
    k, delay = int(k), int(delay)

    info = {"k": k, "Delay": delay}

    if not np.all(np.isfinite(signal)) or np.ptp(signal) == 0:
        warn(
            "The signal has missing values or is constant, which makes it impossible to compute GridEn. Returning np.nan",
            category=NeuroKitWarning,
        )
        info["GDR"] = np.nan
        return np.nan, info

    # Normalization to [0, 1], and section of each value along the axes of the Poincaré plot. The
    # sections are closed on the left (the maximum falls in the last one), and are computed from the
    # signal (rather than from the normalized values), so that the values on the grid lines (e.g.,
    # with integer RR intervals in ms) are assigned exactly
    signal_min, signal_range = np.min(signal), np.ptp(signal)
    sections = np.minimum((signal - signal_min) * k // signal_range, k - 1).astype(int)

    # Poincaré Plot: count of points in each grid (rows for the present, columns for the delayed
    # values)
    hist = np.bincount(sections[:-delay] * k + sections[delay:], minlength=k * k).reshape(k, k)

    # Get frequency
    freq = hist[hist > 0] / hist.sum()

    # Compute Shannon Entropy
    griden, _ = entropy_shannon(freq=freq, **kwargs)

    # Compute Gridded Distribution Rate
    info["GDR"] = np.sum(hist > 0) / hist.size

    if show is True:
        signal_n = (signal - signal_min) / signal_range
        gridlines = np.linspace(0, 1, k + 1)
        _, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(signal_n[:-delay], signal_n[delay:], ".", color="#009688")
        ax1.plot(
            np.tile(gridlines, (2, 1)),
            np.array((np.zeros(k + 1), np.ones(k + 1))),
            color="red",
        )
        ax1.plot(
            np.array((np.zeros(k + 1), np.ones(k + 1))),
            np.tile(gridlines, (2, 1)),
            color="red",
        )
        ax1.plot([0, 1], [0, 1], "k")
        ax1.set_aspect("equal", "box")
        ax1.set_xlabel(r"$X_{i}$")
        ax1.set_ylabel(r"$X_{i + \tau}$")
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        # Number of points in each grid, in the same orientation as the Poincaré plot
        ax2.imshow(hist.T, origin="lower", cmap="rainbow", aspect="equal")
        ax2.set_xlabel(r"$X_{i}$")
        ax2.set_ylabel(r"$X_{i + \tau}$")
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.suptitle("Gridded Poincaré Plot and its Density")

    return griden, info
