# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal


def signal_resample(
    signal,
    desired_length=None,
    sampling_rate=None,
    desired_sampling_rate=None,
    method="interpolation",
):
    """**Resample a continuous signal to a different length or sampling rate**

    Up- or down-sample a signal. The user can specify either a desired length for the vector, or
    input the original sampling rate and the desired sampling rate.

    Parameters
    ----------
    signal :  Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    desired_length : int
        The desired length of the signal.
    sampling_rate : int
        The original sampling frequency (in Hz, i.e., samples/second).
    desired_sampling_rate : int
        The desired (output) sampling frequency (in Hz, i.e., samples/second).
    method : str
        Can be ``"interpolation"`` (see ``scipy.ndimage.zoom()``), ``"numpy"`` for numpy's
        interpolation (see ``np.interp()``),``"pandas"`` for Pandas' time series resampling,
        ``"poly"`` (see ``scipy.signal.resample_poly()``) or ``"FFT"`` (see
        ``scipy.signal.resample()``) for the Fourier method. ``"FFT"`` is the most accurate
        (if the signal is periodic), but becomes exponentially slower as the signal length
        increases. In contrast, ``"interpolation"`` is the fastest, followed by ``"numpy"``,
        ``"poly"`` and ``"pandas"``.

    Returns
    -------
    array
        Vector containing resampled signal values.

    See Also
    --------
    signal_interpolate

    Examples
    --------
    **Example 1**: Downsampling

    .. ipython:: python

      import numpy as np
      import pandas as pd
      import neurokit2 as nk

      signal = nk.signal_simulate(duration=1, sampling_rate=500, frequency=3)

      # Downsample
      data = {}
      for m in ["interpolation", "FFT", "poly", "numpy", "pandas"]:
          data[m] = nk.signal_resample(signal, sampling_rate=500, desired_sampling_rate=30, method=m)

      @savefig p_signal_resample1.png scale=100%
      nk.signal_plot([data[m] for m in data.keys()])
      @suppress
      plt.close()

    **Example 2**: Upsampling

    .. ipython:: python
      :verbatim:

      signal = nk.signal_simulate(duration=1, sampling_rate=30, frequency=3)

      # Upsample
      data = {}
      for m in ["interpolation", "FFT", "poly", "numpy", "pandas"]:
          data[m] = nk.signal_resample(signal, sampling_rate=30, desired_sampling_rate=500, method=m)

      @savefig p_signal_resample2.png scale=100%
      nk.signal_plot([data[m] for m in data.keys()], labels=list(data.keys()))
      @suppress
      plt.close()

    **Example 3**: Benchmark

    .. ipython:: python
      :verbatim:

      signal = nk.signal_simulate(duration=1, sampling_rate=1000, frequency=3)

      # Timing benchmarks
      %timeit nk.signal_resample(signal, method="interpolation",
                                 sampling_rate=1000, desired_sampling_rate=500)
      %timeit nk.signal_resample(signal, method="FFT",
                                 sampling_rate=1000, desired_sampling_rate=500)
      %timeit nk.signal_resample(signal, method="poly",
                                 sampling_rate=1000, desired_sampling_rate=500)
      %timeit nk.signal_resample(signal, method="numpy",
                                 sampling_rate=1000, desired_sampling_rate=500)
      %timeit nk.signal_resample(signal, method="pandas",
                                 sampling_rate=1000, desired_sampling_rate=500)

    """
    if desired_length is None:
        desired_length = int(np.round(len(signal) * desired_sampling_rate / sampling_rate))

    # Sanity checks
    if len(signal) == desired_length:
        return signal

    # Resample
    if method.lower() == "fft":
        resampled = _resample_fft(signal, desired_length)
    elif method.lower() == "poly":
        resampled = _resample_poly(signal, desired_length)
    elif method.lower() == "numpy":
        resampled = _resample_numpy(signal, desired_length)
    elif method.lower() == "pandas":
        resampled = _resample_pandas(signal, desired_length)
    else:
        resampled = _resample_interpolation(signal, desired_length)

    return resampled


# =============================================================================
# Methods
# =============================================================================


def _resample_numpy(signal, desired_length):
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, desired_length, endpoint=False),  # where to interpolate
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal


def _resample_interpolation(signal, desired_length):
    resampled_signal = scipy.ndimage.zoom(signal, desired_length / len(signal))
    return resampled_signal


def _resample_fft(signal, desired_length):
    resampled_signal = scipy.signal.resample(signal, desired_length)
    return resampled_signal


def _resample_poly(signal, desired_length):
    resampled_signal = scipy.signal.resample_poly(signal, desired_length, len(signal))
    return resampled_signal


def _resample_pandas(signal, desired_length):
    # Convert to Time Series
    index = pd.date_range("20131212", freq="ms", periods=len(signal))
    resampled_signal = pd.Series(signal, index=index)

    # Create resampling factor
    resampling_factor = str(np.round(1 / (desired_length / len(signal)), 6)) + "ms"

    # Resample
    resampled_signal = resampled_signal.resample(resampling_factor).bfill().values

    # Sanitize
    resampled_signal = _resample_sanitize(resampled_signal, desired_length)

    return resampled_signal


# =============================================================================
# Internals
# =============================================================================


def _resample_sanitize(resampled_signal, desired_length):
    # Adjust extremities
    diff = len(resampled_signal) - desired_length
    if diff < 0:
        resampled_signal = np.concatenate(
            [resampled_signal, np.full(np.abs(diff), resampled_signal[-1])]
        )
    elif diff > 0:
        resampled_signal = resampled_signal[0:desired_length]
    return resampled_signal
