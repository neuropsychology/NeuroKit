# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import datetime
import scipy.signal
import scipy.ndimage



def signal_resample(signal, desired_length=None, sampling_rate=None, desired_sampling_rate=None, method="interpolation"):
    """Resample a continuous signal.

    This function can be used to up- or down-sample a signal. The user can specify either a desired length for the vector, or input the original sampling rate and the desired sampling rate. See https://github.com/neuropsychology/NeuroKit/scripts/resampling.ipynb for a comparison of the methods.

    Parameters
    ----------
    signal :  list, array or Series
        The signal channel.
    desired_length : int
        The desired length of the signal.
    sampling_rate, desired_sampling_rate : int
        The original and desired (output) sampling frequency (in Hz, i.e., samples/second).
    method : str
        Can be 'numpy' (default) for numpy's interpolation (see `numpy.interp()`), 'pandas' for Pandas' time series resampling, 'interpolation' (see `scipy.ndimage.zoom()`), 'poly' (see `scipy.signal.resample_poly()`) or 'FFT' (see `scipy.signal.resample()`) for the Fourier method. FFT is the most accurate (if the signal is periodic), but becomes exponentially slower as the signal length increases. In contrast, 'numpy' is the fastest, followed by 'poly', 'pandas' and 'interpolation'.

    Returns
    -------
    array
        Array containing resampled signal values.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=20, num=100))
    >>>
    >>> # Downsample
    >>>
    >>> downsampled_interpolation = nk.signal_resample(signal, method="interpolation", sampling_rate=1000, desired_sampling_rate=500)
    >>> downsampled_fft = nk.signal_resample(signal, method="FFT", sampling_rate=1000, desired_sampling_rate=500)
    >>> downsampled_poly = nk.signal_resample(signal, method="poly", sampling_rate=1000, desired_sampling_rate=500)
    >>> downsampled_numpy = nk.signal_resample(signal, method="numpy", sampling_rate=1000, desired_sampling_rate=500)
    >>> downsampled_pandas = nk.signal_resample(signal, method="pandas", sampling_rate=1000, desired_sampling_rate=500)
    >>>
    >>> # Upsample
    >>> upsampled_interpolation = nk.signal_resample(downsampled_interpolation, method="interpolation", sampling_rate=500, desired_sampling_rate=1000)
    >>> upsampled_fft = nk.signal_resample(downsampled_fft, method="FFT", sampling_rate=500, desired_sampling_rate=1000)
    >>> upsampled_poly = nk.signal_resample(downsampled_poly, method="poly", sampling_rate=500, desired_sampling_rate=1000)
    >>> upsampled_numpy = nk.signal_resample(downsampled_numpy, method="numpy", sampling_rate=500, desired_sampling_rate=1000)
    >>> upsampled_pandas = nk.signal_resample(downsampled_pandas, method="pandas", sampling_rate=500, desired_sampling_rate=1000)
    >>>
    >>> # Compare with original
    >>> pd.DataFrame({"Original": signal,
                      "Interpolation": upsampled_interpolation,
                      "FFT": upsampled_fft,
                      "Poly": upsampled_poly,
                      "Numpy": upsampled_numpy,
                      "Pandas": upsampled_pandas}).plot(style='.-')
    >>>
    >>> # Timing benchmarks
    >>> %timeit nk.signal_resample(signal, method="interpolation", sampling_rate=1000, desired_sampling_rate=500)
    >>> %timeit nk.signal_resample(signal, method="FFT", sampling_rate=1000, desired_sampling_rate=500)
    >>> %timeit nk.signal_resample(signal, method="poly", sampling_rate=1000, desired_sampling_rate=500)
    >>> %timeit nk.signal_resample(signal, method="numpy", sampling_rate=1000, desired_sampling_rate=500)
    >>> %timeit nk.signal_resample(signal, method="pandas", sampling_rate=1000, desired_sampling_rate=500)

    See Also
    --------
    scipy.signal.resample_poly, scipy.signal.resample, scipy.ndimage.zoom
    """
    if desired_length is None:
        desired_length = round(len(signal) * desired_sampling_rate / sampling_rate)

    # Sanity checks
    if len(signal) == desired_length:
        return(signal)

    # Resample
    if method.lower() == "fft":
        resampled = _resample_fft(signal, desired_length)
    elif method.lower() == "poly":
        resampled =  _resample_poly(signal, desired_length)
    elif method.lower() == "numpy":
        resampled =  _resample_numpy(signal, desired_length)
    elif method.lower() == "pandas":
        resampled =  _resample_pandas(signal, desired_length)
    else:
        resampled =  _resample_interpolation(signal, desired_length)

    return(resampled)




# =============================================================================
# Internals
# =============================================================================

def _resample_numpy(signal, desired_length):
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, desired_length, endpoint=False),  # where to interpolate
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return(resampled_signal)


def _resample_interpolation(signal, desired_length):
    resampled_signal = scipy.ndimage.zoom(signal, desired_length/len(signal))
    return(resampled_signal)


def _resample_fft(signal, desired_length):
    resampled_signal = scipy.signal.resample(signal, desired_length)
    return(resampled_signal)


def _resample_poly(signal, desired_length):
    resampled_signal = scipy.signal.resample_poly(signal, desired_length, len(signal))
    return(resampled_signal)


def _resample_pandas(signal, desired_length):
    # Convert to Time Series
    index = pd.date_range('20131212', freq='L', periods=len(signal))
    resampled_signal = pd.Series(signal, index=index)

    # Create resampling factor
    resampling_factor = str(1/(desired_length / len(signal))) + "L"

    # Resample
    resampled_signal = resampled_signal.resample(resampling_factor).bfill().values

    # Adjust extremities
    diff = len(resampled_signal) - desired_length
    if diff < 0:
        resampled_signal = np.concatenate([resampled_signal, np.full(abs(diff), resampled_signal[-1])])
    elif diff > 0:
        resampled_signal = resampled_signal[0:desired_length]

    return(resampled_signal)