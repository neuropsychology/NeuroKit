import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .entropy_shannon import entropy_shannon


def entropy_grid(signal, delay=1, k=3, show=False, **kwargs):
    """**Grid Entropy (GridEn)**

    Grid Entropy (GridEn or GDEn) is defined as a gridded descriptor of a :func:`Poincaré plot <.hrv_nonlinear>`,
    which is a two-dimensional phase space diagram of a time series that plots the present sample
    of a time series with respect to their delayed values. The plot is divided into :math:`n*n`
    grids, and the :func:`Shannon entropy <entropy_shannon>` is computed from the probability
    distribution of the number of points in each grid.

    Yan et al. (2019) define two novel measures, namely **GridEn** and **Gridded Distribution Rate
    (GDR)**, the latter being the percentage of grids containing points.


    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    k : int
        The number of sections that the Poincaré plot is divided into. It is a coarse
        graining parameter that defines how fine the grid is.
    show : bool
        Plot the Poincaré plot.
    **kwargs : optional
        Other keyword arguments, such as the logarithmic ``base`` to use for
        :func:`entropy_shannon`.

    Returns
    -------
    griden : float
        Grid Entropy of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used.

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
      phasen, info = nk.entropy_grid(signal, k=3, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      phasen

      @savefig p_entropy_grid2.png scale=100%
      phasen, info = nk.entropy_grid(signal, k=10, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      info["GDR"]

    References
    ----------
    * Yan, C., Li, P., Liu, C., Wang, X., Yin, C., & Yao, L. (2019). Novel gridded descriptors of
      poincaré plot for analyzing heartbeat interval time-series. Computers in biology and
      medicine, 109, 280-289.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    info = {"k": k, "Delay": delay}

    # Normalization
    Sig_n = (signal - min(signal)) / np.ptp(signal)

    # Poincaré Plot
    Temp = np.array([Sig_n[:-delay], Sig_n[delay:]])

    # Get count of points in each grid
    hist, _, _ = np.histogram2d(Temp[0, :], Temp[1, :], k)

    # Get frequency
    freq = np.flipud(hist.T) / hist.sum()
    freq = freq[freq > 0]

    # Compute Shannon Entropy
    griden, _ = entropy_shannon(freq=freq, **kwargs)

    # Compute Gridded Distribution Rate
    info["GDR"] = np.sum(hist != 0) / hist.size

    if show is True:

        gridlines = np.linspace(0, 1, k + 1)
        plt.subplots(1, 2)
        x1 = plt.subplot(121)
        ax1 = plt.axes(x1)

        ax1.plot(Sig_n[:-delay], Sig_n[delay:], ".", color="#009688")
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
        ax1.set_ylabel(r"$X_{i} + \tau$")
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        x2 = plt.subplot(122)
        ax2 = plt.axes(x2)
        ax2.imshow(np.fliplr(hist), cmap="rainbow", aspect="equal")
        ax1.set_xlabel(r"$X_{i}$")
        ax1.set_ylabel(r"$X_{i} + \tau$")
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.suptitle("Gridded Poincaré Plot and its Density")

    return griden, info
