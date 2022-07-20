# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..signal import signal_detrend


def fractal_sda(signal, scales=None, show=False):
    """**Standardised Dispersion Analysis (SDA)**

    SDA is part of a family of dispersion techniques used to compute fractal dimension.
    The standardized time series is divided in bins of different sizes and their standard deviation
    (SD) is calculated. The relationship between the SD and the bin size can be an indication
    of the presence of power-laws. For instance, if the SD systematically increases or
    decreases with larger bin sizes, this means the fluctuations depend on the size of the bins.
    The dispersion measurements are in units of the standard error of the mean. An FD of 1.5
    indicates random data series, while values approaching 1.20 indicate 1/f scaling.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    scales : list
        The scales at which the signal is binned for evaluating the dispersions. If not ``None``, it
        should be a list of integer powers of 2 (e.g., scales = [1, 2, 4, 8, 16...]) including 1
        (meaning that the data points are treated individually).
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
      inference based
    on empirical waveforms. Frontiers in Physiology, 4, 75. https://doi.org/10.3389/fphys.2013.00075
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
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Detrend signal
    signal = signal_detrend(signal)

    # compute SD using population formula N instead of usual bias corrected N-1
    signal = (signal - np.nanmean(signal)) / np.nanstd(signal)

    n = len(signal)

    # Set scales to be an integer power of 2
    if scales is None:
        scale_min = 1
        scale_max = int(np.floor(np.log2(n / 2)))
        scales = np.append(1, 2 ** np.arange(scale_min, scale_max + 1))  # include binsize = 1 too

    # sanitize scales
    scales = scales[scales <= n / 2]

    # Assess variability using the SD of means of progressively larger adjacent samples
    sds = np.zeros(len(scales))
    for i, scale in enumerate(scales):
        max_n = int(len(signal) / scale) * scale
        splits = np.split(signal[0:max_n], scale)
        # compute sd of the sampling distribution of means (mean of each bin)
        if scale == 1:
            sds[i] = np.std(splits)
            # sd of original standardized time series is 1
        else:
            sds[i] = np.std([np.mean(split) for split in splits])

    # Get slope
    slope, intercept = np.polyfit(np.log10(scales), np.log10(sds), 1)
    if show:
        _fractal_sda_plot(sds, scales, slope, intercept, ax=None)

    # FD = 1 - slope
    return 1 - slope, {"Slope": slope, "SD": sds, "Scale": scales}


def _fractal_sda_plot(sds, scales, slope, intercept, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle(
            "Standardized Dispersion as a function of Sample-Bin size"
            + ", slope = "
            + str(np.round(slope, 2))
        )
    else:
        fig = None
        ax.set_title(
            "Standardized Dispersion as a function of Sample-Bin size"
            + ", slope = "
            + str(np.round(slope, 2))
        )

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
