import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..signal import signal_detrend


def fractal_sda(signal, scales=None, show=False):
    """**Standardised Dispersion Analysis (SDA)**

    SDA is part of a family of dispersion techniques used to compute fractal dimension.
    The standardized time series is divided in adjacent, non-overlapping bins of a given size, and
    the standard deviation (SD) of the bin means is calculated. This is repeated for different bin
    sizes. The relationship between the SD and the bin size can be an indication
    of the presence of power-laws. For instance, if the SD systematically increases or
    decreases with larger bin sizes, this means the fluctuations depend on the size of the bins.
    The dispersion measurements are in units of the standard error of the mean. The FD is
    :math:`1 - slope` of the log-log relationship (i.e., :math:`2 - H` for fractional Gaussian
    noises): an FD of 1.5 indicates random data series (the SD of the means then decreases as
    :math:`1/\\sqrt{size}`), while values approaching 1.20 indicate 1/f scaling (the limit
    H = 1, for which the finite length of the signal keeps the estimate above 1).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    scales : list
        The bin sizes (number of data points per bin) at which the dispersions are evaluated. If
        ``None`` (default), uses the integer powers of 2 up to half the signal's length, including
        1 (meaning that the data points are treated individually), i.e., ``[1, 2, 4, 8, ...]``.

        .. note:: This default range (as in Hasselman's *casnet*) leaves only 2 bins at the
           largest scale, whose SD is very noisy and biased downward in log-space. Stopping at the
           scale that still has 8 bins (``n / 8``) reduces the error of the FD (e.g., an RMSE of
           0.041 instead of 0.100 for white noise of 1,000 samples). We might change the default
           in the future.
    show : bool
        If ``True``, returns the log-log plot of standardized dispersion versus bin size.

    Returns
    ----------
    sda : float
        Estimate of the fractal dimension using the conversion formula of
        SDA (Hasselman, 2013).
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute SDA.

    References
    ----------
    * https://complexity-methods.github.io/book/standardised-dispersion-analysis-sda.html
    * Hasselman, F. (2013). When the blind curve is finite: dimension estimation and model
      inference based on empirical waveforms. Frontiers in Physiology, 4, 75.
      https://doi.org/10.3389/fphys.2013.00075
    * Holden, J. G. (2005). Gauging the fractal dimension of response times from cognitive tasks.
      Contemporary nonlinear methods for behavioral scientists: A webbook tutorial, 267-318.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=6, sampling_rate=200, frequency=[5, 6], noise=0.5)

      @savefig p_fractal_sda.png scale=100%
      sda, _ = nk.fractal_sda(signal, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      sda


    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    # Detrend signal
    signal = signal_detrend(np.asarray(signal, dtype=float))

    # compute SD using population formula N instead of usual bias corrected N-1
    signal = (signal - np.nanmean(signal)) / np.nanstd(signal)

    n = len(signal)

    # Set scales to be an integer power of 2 (including binsize = 1), up to n / 2
    # TODO: consider stopping at n / 8 (at least 8 bins per scale): with only 2 bins, the largest
    # scale's SD is noisy and biased downward in log-space (white noise of 1,000 samples: FD 1.557
    # with an RMSE of 0.100 with n / 2, 1.513 and 0.041 with n / 8; see the `scales` docstring).
    if scales is None:
        scales = 2 ** np.arange(0, int(np.floor(np.log2(n / 2))) + 1)

    # sanitize scales (at least 2 bins are needed to compute a dispersion)
    scales = np.unique(np.asarray(scales, dtype=int))
    scales = scales[(scales >= 1) & (scales <= n / 2)]
    if len(scales) < 2:
        raise ValueError("NeuroKit error: fractal_sda(): at least 2 valid scales (between 1 and n/2) are needed.")

    # Assess variability using the SD of means of progressively larger adjacent bins
    sds = np.zeros(len(scales))
    for i, scale in enumerate(scales):
        max_n = (n // scale) * scale
        # means of the n // scale adjacent bins of `scale` points each
        means = np.nanmean(signal[0:max_n].reshape(-1, scale), axis=1)
        # SD of the sampling distribution of the means (sample formula, as the number of bins can
        # be as small as 2)
        sds[i] = np.nanstd(means, ddof=1)

    # Get slope
    slope, intercept = np.polyfit(np.log10(scales), np.log10(sds), 1)
    if show:
        _fractal_sda_plot(sds, scales, slope, intercept, ax=None)

    # FD = 1 - slope
    return 1 - slope, {"Slope": slope, "SD": sds, "Scale": scales}


def _fractal_sda_plot(sds, scales, slope, intercept, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle("Standardized Dispersion as a function of Sample-Bin size" + ", slope = " + str(np.round(slope, 2)))
    else:
        fig = None
        ax.set_title("Standardized Dispersion as a function of Sample-Bin size" + ", slope = " + str(np.round(slope, 2)))

    ax.set_ylabel(r"$\log_{10}$(Standardized Dispersion)")
    ax.set_xlabel(r"$\log_{10}$(Bin Size)")

    ax.scatter(np.log10(scales), np.log10(sds), marker="o", zorder=2)

    fit_values = [slope * i + intercept for i in np.log10(scales)]
    ax.plot(
        np.log10(scales),
        fit_values,
        color="#FF9800",
        zorder=1,
        label="Fractal Dimension = " + str(np.round(1 - slope, 2)),
    )
    ax.legend(loc="lower right")
