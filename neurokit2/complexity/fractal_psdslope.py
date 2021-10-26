import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..signal import signal_detrend, signal_psd


def fractal_psdslope(signal, sampling_rate=1000, show=True, **kwargs):
    """Fractal dimension via Power Spectral Density (PSD) slope

    Power Spectral Density slope (PSDslope) analysis first transforms the time series into the
    frequency domain, and breaks down the signal into sine and cosine waves of a particular amplitude
    that together "add-up" to represent the original signal. If there is a systematic relationship
    between the frequencies in the signal and the power of those frequencies, this will reveal itself
    in log-log coordinates as a linear relationship. The slope of the best fitting line is taken as an
    estimate of the scaling exponent and can be converted to an estimate of the fractal dimension.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    show : bool
        If True, returns the log-log plot of PSD versus frequency.
    **kwargs
        Other arguments to be passed to ``signal_psd()`` (such as 'method').

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)
    >>>
    >>> psdslope, info = nk.fractal_psdslope(signal, sampling_rate=200)
    >>> psdslope

    References
    ----------
    - https://complexity-methods.github.io/book/power-spectral-density-psd-slope.html

    - Hasselman, F. (2013). When the blind curve is finite: dimension estimation and model inference
    based on empirical waveforms. Frontiers in Physiology, 4, 75. https://doi.org/10.3389/fphys.2013.00075
    """

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Translated from https://github.com/FredHasselman/casnet/blob/master/R/fd.R
    # Detrend
    signal = signal_detrend(signal)

    # Standardise using N instead of N-1
    signal = (signal - np.nanmean(signal)) / np.nanstd(signal)

    # Get PSD
    psd = signal_psd(signal, sampling_rate=sampling_rate, show=False, **kwargs)
    psd = psd[psd["Frequency"] > 0]

    psd["Frequency_Norm"] = psd["Frequency"] / sampling_rate
    psd["Size"] = psd["Frequency"]
    psd["Bulk"] = 2 * psd["Power"]
    
    # TODO: add log-log plot
    # plot(x=log2(psd$freq), y=log2(psd$spec*2),pch=".")

    # First check the global slope for anti-persistent noise (GT +0.20)
    # If so, fit the line starting from the highest frequency
    slope, intercept = np.polyfit(np.log(psd["Size"]), np.log(psd["Bulk"]), 1)

    if slope > 0.2:
        slope, intercept = np.polyfit(np.log(np.flip(psd["Size"])), np.log(np.flip(psd["Bulk"])), 1)

    # Convert from periodogram based self-affinity parameter estimate (`sa`) to an informed estimate of the (fractal) dimension (FD). See Hassleman (2013).
    fd = 3 / 2 + ((14 / 33) * np.tanh(slope * np.log(1 + np.sqrt(2))))

    # Plot
    if show:
        _fractal_psdslope_plot(psd["Size"], psd["Bulk"], slope, intercept, fd, ax=None)

    return fd, {"Sampling_Rate": sampling_rate, "PSD": psd}


def _fractal_psdslope_plot(frequency, psd, slope, intercept, fd, ax=None):

    ax = None
    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle("Power Spectral Density (PSD) slope analysis" +
                     ", slope = " + str(np.round(slope, 2)))
    else:
        fig = None
        ax.set_title("Power Spectral Density (PSD) slope analysis" +
                     ", slope = " + str(np.round(slope, 2)))

    ax.set_ylabel(r"$ln$(PSD)")
    ax.set_xlabel(r"$ln$(Frequency)")

    ax.scatter(np.log10(frequency), np.log10(psd),
               marker="o", zorder=2)

    fit_values = [slope * i + intercept for i in np.log10(frequency)]
    ax.plot(np.log10(frequency), fit_values, color="#FF9800", zorder=1,
            label="Fractal Dimension = " + str(np.round(fd, 2)))
    ax.legend(loc="lower right")

    return fig
