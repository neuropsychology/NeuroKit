import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..signal import signal_detrend, signal_psd


def fractal_psdslope(signal, method="voss1988", show=False, **kwargs):
    """**Fractal dimension via Power Spectral Density (PSD) slope**

    Fractal exponent can be computed from Power Spectral Density slope (PSDslope) analysis in
    signals characterized by a frequency power-law dependence.

    It first transforms the time series into the frequency domain, and breaks down the signal into
    sine and cosine waves of a particular amplitude that together "add-up" to represent the
    original signal.
    If there is a systematic relationship between the frequencies in the signal and the power of
    those frequencies, this will reveal itself in log-log coordinates as a linear relationship. The
    slope of the best fitting line is taken as an estimate of the fractal scaling exponent and can
    be converted to an estimate of the fractal dimension.

    A slope of 0 is consistent with white noise, and a slope of less than 0 but greater than -1,
    is consistent with pink noise i.e., 1/f noise. Spectral slopes as steep as -2 indicate
    fractional Brownian motion, the epitome of random walk processes.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        Method to estimate the fractal dimension from the slope,
        can be ``"voss1988"`` (default) or ``"hasselman2013"``.
    show : bool
        If True, returns the log-log plot of PSD versus frequency.
    **kwargs
        Other arguments to be passed to ``signal_psd()`` (such as ``method``).

    Returns
    ----------
    slope : float
        Estimate of the fractal dimension obtained from PSD slope analysis.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to perform PSD slope analysis.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal with Laplace Noise
      signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)

      # Compute the Fractal Dimension from PSD slope
      @savefig p_fractal_psdslope1.png scale=100%
      psdslope, info = nk.fractal_psdslope(signal, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      psdslope

    References
    ----------
    * https://complexity-methods.github.io/book/power-spectral-density-psd-slope.html
    * Hasselman, F. (2013). When the blind curve is finite: dimension estimation and model
      inference based on empirical waveforms. Frontiers in Physiology, 4, 75. https://doi.org/10.3389/fphys.2013.00075
    * Voss, R. F. (1988). Fractals in nature: From characterization to simulation. The Science of
      Fractal Images, 21-70.
    * Eke, A., Hermán, P., Kocsis, L., and Kozak, L. R. (2002). Fractal characterization of
      complexity in temporal physiological signals. Physiol. Meas. 23, 1-38.

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

    # Get psd with fourier transform
    psd = signal_psd(signal, sampling_rate=1000, method="fft", show=False, **kwargs)
    psd = psd[psd["Frequency"] < psd.quantile(0.25).iloc[0]]
    psd = psd[psd["Frequency"] > 0]

    # Get slope
    slope, intercept = np.polyfit(np.log10(psd["Frequency"]), np.log10(psd["Power"]), 1)

    # "Check the global slope for anti-persistent noise (GT +0.20) and fit the line starting from
    # the highest frequency" in FredHasselman/casnet.
    # Not sure about that, commenting it out for now.
    # if slope > 0.2:
    #     slope, intercept = np.polyfit(np.log10(np.flip(psd["Frequency"])), np.log10(np.flip(psd["Power"])), 1)

    # Sanitize method name
    method = method.lower()
    if method in ["voss", "voss1988"]:
        fd = (5 - slope) / 2
    elif method in ["hasselman", "hasselman2013"]:
        # Convert from periodogram based self-affinity parameter estimate (`sa`) to an informed
        # estimate of fd
        fd = 3 / 2 + ((14 / 33) * np.tanh(slope * np.log(1 + np.sqrt(2))))

    if show:
        _fractal_psdslope_plot(psd["Frequency"], psd["Power"], slope, intercept, fd, ax=None)

    return fd, {"Slope": slope, "Method": method}


# =============================================================================
# Plotting
# =============================================================================
def _fractal_psdslope_plot(frequency, psd, slope, intercept, fd, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle(
            "Power Spectral Density (PSD) slope analysis" + ", slope = " + str(np.round(slope, 2))
        )
    else:
        fig = None
        ax.set_title(
            "Power Spectral Density (PSD) slope analysis" + ", slope = " + str(np.round(slope, 2))
        )

    ax.set_ylabel(r"$\log_{10}$(Power)")
    ax.set_xlabel(r"$\log_{10}$(Frequency)")
    # ax.scatter(np.log10(frequency), np.log10(psd), marker="o", zorder=1)
    ax.plot(np.log10(frequency), np.log10(psd), zorder=1)

    # fit_values = [slope * i + intercept for i in np.log10(frequency)]
    fit = np.polyval((slope, intercept), np.log10(frequency))
    ax.plot(
        np.log10(frequency),
        fit,
        color="#FF9800",
        zorder=2,
        label="Fractal Dimension = " + str(np.round(fd, 2)),
    )
    ax.legend(loc="lower right")

    return fig
