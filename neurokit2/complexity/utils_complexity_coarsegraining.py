# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from ..signal import signal_interpolate
from .utils_complexity_embedding import complexity_embedding


def complexity_coarsegraining(
    signal, scale=2, method="nonoverlapping", show=False, **kwargs
):
    """**Coarse-graining of a signal**

    The goal of coarse-graining is to represent the signal at a different "scale". The
    coarse-grained time series for a scale factor Tau (:math:`\\tau`) are obtained by averaging
    non-overlapping windows of size Tau. In most of the complexity metrics, multiple coarse-grained
    segments are constructed for a given signal, to represent the signal at different scales (hence
    the "multiscale" adjective).

    .. figure:: ../img/wu2013a.png
       :alt: Figure from Wu et al. (2013).
       :target: https://doi.org/10.1016/j.physleta.2014.03.034

    This coarse-graining procedure is similar to moving averaging and the decimation of the original
    time series. The length of each coarse-grained time series is N/Tau. For ``scale = 1``, the
    coarse-grained time series is simply the original time series itself.

    The coarse graining procedure (used for instance in MSE) is considered a shortcoming that
    decreases the entropy rate artificially (Nikulin, 2004). One of the core issue is that the
    length of coarse-grained signals becomes smaller as the scale increases.

    To address this issue of length, several methods have been proposed, such as **adaptive
    resampling** (Liu et al. 2012), **moving average** (Wu et al. 2013), or **timeshift**
    (Wu et al. 2013).

    * **Non-overlapping** (default): The coarse-grained time series are constructed by averaging
      non-overlapping windows of given size.
    * **Interpolate**: Interpolates (i.e., resamples) the coarse-grained time series to match the
      original signal length (currently using a monotonic cubic method, but let us know if you have
      any opinion on that).
    * **Moving average**: The coarse-grained time series via a moving average.
    * **Time-shift**: For each scale, a *k* number of coarse-grained vectors are constructed (see
      **Figure** below). Somewhat similar to moving-average, with the difference that the time lag
      creates new vectors.

    .. figure:: ../img/wu2013b.png
       :alt: Figure from Wu et al. (2013).
       :target: https://doi.org/10.1016/j.physleta.2014.03.034

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    scale : int
        The size of the windows that the signal is divided into. Also referred to as Tau
        :math:`\\tau`, it represents the scale factor and corresponds to
        the amount of coarsegraining.
    method : str
        Can be ``"nonoverlapping"``, ``"rolling"``, ``"interpolate"``, or ``"timeshift"``.
    show : bool
        If ``True``, will show the coarse-grained signal.
    **kwargs
        Other arguments (not used currently).

    Returns
    -------
    array
        The coarse-grained signal.

    See Also
    ------------
    complexity_delay, complexity_dimension

    Examples
    ---------

    **Simple examples**
    .. ipython:: python

      import neurokit2 as nk

      signal = [0, 2, 4, 6, 8, 10]
      nk.complexity_coarsegraining(signal, scale=2)

      signal = [0, 1, 2, 0, 1]
      nk.complexity_coarsegraining(signal, scale=3)

      nk.complexity_coarsegraining(signal=range(10), method="interpolate")
      nk.complexity_coarsegraining(signal=range(10), method="rolling")

    **Simulated signal**
    .. ipython:: python

      signal = nk.signal_simulate(duration=2, frequency=[5, 20])

      @savefig p_complexity_coarsegraining1.png scale=100%
      coarsegrained = nk.complexity_coarsegraining(signal, scale=40, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_coarsegraining2.png scale=100%
      coarsegrained = nk.complexity_coarsegraining(signal, scale=40, method="interpolate", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_coarsegraining3.png scale=100%
      coarsegrained = nk.complexity_coarsegraining(signal, scale=40, method="rolling", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      signal = nk.signal_simulate(duration=0.5, frequency=[5, 20])

      @savefig p_complexity_coarsegraining4.png scale=100%
      coarsegrained = nk.complexity_coarsegraining(signal, scale=30, method="timeshift", show=True)
      @suppress
      plt.close()

    **Benchmarking**
    .. ipython:: python

      signal = nk.signal_simulate(duration=10, frequency=5)
      scale = 2
      x_pd = pd.Series(signal).rolling(window=scale).mean().values[scale-1::scale]
      x_nk = nk.complexity_coarsegraining(signal, scale=scale)
      np.allclose(x_pd - x_nk, 0)

      %timeit x_pd = pd.Series(signal).rolling(window=scale).mean().values[scale-1::scale]
      %timeit x_nk = nk.complexity_coarsegraining(signal, scale=scale)

      signal = nk.signal_simulate(duration=30, frequency=5)
      scale = 3

      x_pd = pd.Series(signal).rolling(window=scale).mean().values[scale-1::]
      x_nk = nk.complexity_coarsegraining(signal, scale=scale, rolling=True)
      np.allclose(x_pd - x_nk[1:-1], 0)

      %timeit pd.Series(signal).rolling(window=scale).mean().values[scale-1::]
      %timeit nk.complexity_coarsegraining(signal, scale=scale, rolling=True)

    References
    -----------
    * Su, C., Liang, Z., Li, X., Li, D., Li, Y., & Ursino, M. (2016). A comparison of multiscale
      permutation entropy measures in on-line depth of anesthesia monitoring. PLoS One, 11(10),
      e0164104.
    * Nikulin, V. V., & Brismar, T. (2004). Comment on "Multiscale entropy analysis of complex
      physiologic time series”" Physical review letters, 92(8), 089803.
    * Liu, Q., Wei, Q., Fan, S. Z., Lu, C. W., Lin, T. Y., Abbod, M. F., & Shieh, J. S. (2012).
      Adaptive computation of multiscale entropy and its application in EEG signals for monitoring
      depth of anesthesia during surgery. Entropy, 14(6), 978-992.
    * Wu, S. D., Wu, C. W., Lee, K. Y., & Lin, S. G. (2013). Modified multiscale entropy for
      short-term time series analysis. Physica A: Statistical Mechanics and its Applications, 392
      (23), 5865-5873.
    * Wu, S. D., Wu, C. W., Lin, S. G., Wang, C. C., & Lee, K. Y. (2013). Time series analysis
      using composite multiscale entropy. Entropy, 15(3), 1069-1084.

    """
    # Sanity checks
    if scale in [0, 1]:
        return signal
    n = len(signal)
    if scale > n:
        return np.array([])

    if method in ["nonoverlapping", "resampling", "interpolate"]:
        # The following is a fast alternative to:
        # pd.Series(signal).rolling(window=scale).mean().values[scale-1::scale]

        # Get max j
        j = n // scale
        # Coarse-grain
        coarse = np.nanmean(np.reshape(signal[0 : j * scale], (j, scale)), axis=1)

        if method in ["resampling", "interpolate"]:
            x_values = (np.arange(len(coarse)) * scale + scale / 2).astype(int)
            coarse = signal_interpolate(
                x_values, coarse, x_new=np.arange(n), method="monotone_cubic"
            )

    elif method == "rolling":
        # See https://github.com/neuropsychology/NeuroKit/pull/892
        coarse = complexity_embedding(signal, dimension=scale, delay=1).mean(axis=1)

    elif method == "timeshift":
        coarse = np.transpose(
            np.reshape(signal[: scale * (n // scale)], (n // scale, scale))
        )

    else:
        raise ValueError("Unknown `method`: {}".format(method))

    if show is True:
        _complexity_coarsegraining_show(signal[0:n], coarse, method=method)
    return coarse


# =============================================================================
# Utils
# =============================================================================
def _complexity_coarsegraining_show(signal, coarse, method="nonoverlapping"):
    plt.plot(signal, linewidth=1.5)
    if method == "nonoverlapping":
        plt.plot(
            np.linspace(0, len(signal), len(coarse)),
            coarse,
            color="red",
            linewidth=0.75,
        )
        plt.scatter(
            np.linspace(0, len(signal), len(coarse)), coarse, color="red", linewidth=0.5
        )
    elif method == "timeshift":
        for i in range(len(coarse)):
            plt.plot(
                np.arange(i, len(signal) - len(coarse) + i + 1, len(coarse)),
                coarse[i],
                color="red",
                linewidth=0.75,
            )
    else:
        plt.plot(
            np.linspace(0, len(signal), len(coarse)), coarse, color="red", linewidth=1
        )
    plt.title(f'Coarse-graining using method "{method}"')


# =============================================================================
# Get Scale Factor
# =============================================================================
def _get_scales(signal, scale="default", dimension=2):
    """Select scale factors"""
    if scale is None or scale == "max":
        scale = np.arange(1, len(signal) // 2)  # Set to max
    elif scale == "default":
        # See https://github.com/neuropsychology/NeuroKit/issues/75#issuecomment-583884426
        scale = np.arange(1, int(len(signal) / (dimension + 10)))
    elif isinstance(scale, int):
        scale = np.arange(1, scale + 1)

    return scale
