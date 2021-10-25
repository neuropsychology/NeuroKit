import numpy as np
import pandas as pd

from ..signal import signal_detrend


def fractal_sda(signal):
    """Standardised Dispersion Analysis (SDA)

    The standardised time series is divided in bins of different sizes and their standard deviation (SD) is calculated. The relationship between the SD and the bin size can be an indication of the presence of power-laws. For instance, if the SD systematically increases or decreases with larger bin sizes, this means the fluctuations depend on the size of the bins. See `Hasselman (2019) <https://complexity-methods.github.io/book/standardised-dispersion-analysis-sda.html>`_.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    References
    ----------
    - https://complexity-methods.github.io/book/standardised-dispersion-analysis-sda.html
    - Hasselman, F. (2013). When the blind curve is finite: dimension estimation and model inference based on empirical waveforms. Frontiers in Physiology, 4, 75. https://doi.org/10.3389/fphys.2013.00075

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)
    >>> sda, _ = nk.fractal_sda(signal)
    >>> sda

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Detrend signal
    signal = signal_detrend(signal)

    # Standardise using N instead of N-1
    signal = (signal - np.nanmean(signal)) / np.nanstd(signal)

    n = len(signal)
    if n < 32:
        raise ValueError("At least 32 data points of signal are required for this function.")
    else:
        scale_resolution = 30

    # standardise using N instead of N-1.
    scale_min = 2
    scale_max = int(np.floor(np.log2(n / 2)))
    scales = np.arange(scale_min, scale_max, (scale_max - scale_min) / scale_resolution)
    scales = np.append(scales, [scale_max])
    scales = np.unique(np.power(2, scales).astype(int))

    # sanitize scales
    scales = scales[scales <= n / 2]

    sds = np.zeros(len(scales))
    for i, scale in enumerate(scales):
        max_n = int(len(signal) / scale) * scale
        splits = np.split(signal[0:max_n], scale)
        sds[i] = np.mean([np.std(split) for split in splits])

    # Get slope
    _, slope = np.polyfit(np.log(scales), np.log(sds), 1)

    # Convert Standardised Dispersion Analysis (SDA) estimate of self-affinity parameter (`SA`) to an informed estimate of the fractal dimension (FD). See Hassleman (2013).
    sda = 1 - slope
    return sda, {"SD": sds, "Scale": scales}
