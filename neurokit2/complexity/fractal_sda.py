# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..signal import signal_detrend
from ..stats import standardize


def fractal_sda(signal, robust=False, show=True):
    """Standardised Dispersion Analysis (SDA)
    
    SDA is part of a family of dispersion techniques used to compute fractal dimension.
    The standardised time series is divided in bins of different sizes and their standard deviation (SD)
    is calculated. The relationship between the SD and the bin size can be an indication
    of the presence of power-laws. For instance, if the SD systematically increases or
    decreases with larger bin sizes, this means the fluctuations depend on the size of the bins.
    The dispersion measurements are in units of the standard error of the mean.

    See `Hasselman (2019) <https://complexity-methods.github.io/book/standardised-dispersion-analysis-sda.html>`_.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    robust : bool
        If True, centering is done by substracting the median from the variables and dividing it by
        the median absolute deviation (MAD). If False, variables are standardized by substracting the
        mean and dividing it by the standard deviation (SD).
    show : bool
        If True, returns the log-log plot of standardized dispersion versus bin size.

    References
    ----------
    - Hasselman, F. (2013). When the blind curve is finite: dimension estimation and model inference based
    on empirical waveforms. Frontiers in Physiology, 4, 75. https://doi.org/10.3389/fphys.2013.00075

    - Holden, J. G. (2005). Gauging the fractal dimension of response times from cognitive tasks.
    Contemporary nonlinear methods for behavioral scientists: A webbook tutorial, 267-318.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=6, sampling_rate=200, frequency=[5, 6], noise=0.5)
    >>> sda, _ = nk.fractal_sda(signal, show=False)
    >>> sda  #doctest: +SKIP

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Detrend signal
    signal = signal_detrend(signal)

    # Standardize
    if not robust:
        # compute SD using population formula N instead of usual bias corrected N-1
        signal = (signal - np.nanmean(signal)) / np.nanstd(signal)
    else:
        signal = standardize(signal, robust=True)

    n = len(signal)
    # Must be at least 2 bins of size 512
    if n < 1024:
        raise ValueError(
            "NeuroKit error: fractal_sda(): At least 1024 data points of signal are required."
            )
    else:
        scale_resolution = 30

    # Set scales
    scale_min = 2
    scale_max = int(np.floor(np.log2(n / 2)))
    scales = np.arange(scale_min, scale_max, (scale_max - scale_min) / scale_resolution)
    scales = np.append(scales, [scale_max])
    scales = np.unique(np.power(2, scales).astype(int))

    # sanitize scales
    scales = scales[scales <= n / 2]

    # Assess variability using the SD of means of progressively larger adjacent samples
    sds = np.zeros(len(scales))
    for i, scale in enumerate(scales):
        max_n = int(len(signal) / scale) * scale
        splits = np.split(signal[0:max_n], scale)
        # compute sd of the sampling distribution of means (mean of each bin)
        sds[i] = np.std([np.mean(split) for split in splits])

    # Get slope
    slope, intercept = np.polyfit(np.log10(scales), np.log10(sds), 1)
    if show:
        _fractal_dfa_plot(sds, scales, slope, intercept, ax=None)

    # FD = 1 - slope
    return 1 - slope, {"SD": sds, "Scale": scales}


def _fractal_dfa_plot(sds, scales, slope, intercept, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle("Standardized Dispersion as a function of Sample-Bin size" +
                     ", slope = " + str(np.round(slope, 2)))
    else:
        fig = None
        ax.set_title(
            "Standardized Dispersion as a function of Sample-Bin size"
            + ", slope = "
            + str(np.round(slope, 2))
        )

    ax.set_ylabel(r"$log10$(Standardized Dispersion)")
    ax.set_xlabel(r"$log10$(Bin Size)")

    ax.scatter(np.log10(scales), np.log10(sds),
               marker="o", zorder=2)

    fit_values = [slope * i + intercept for i in np.log10(scales)]
    ax.plot(np.log10(scales), fit_values, color="#FF9800", zorder=1,
            label="Fractal Dimension = " + str(np.round(1 - slope, 2)))
    ax.legend(loc="lower right")

    return fig
