# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.signal



def signal_resample(signal, desired_length=None, sampling_rate=None, desired_sampling_rate=None, method="interpolation"):
    """Resample a continuous signal.

    This function can be used to up- or down-sample a signal. The user can specify either a desired length for the vector, or input the original sampling rate and the desired sampling rate.

    Parameters
    ----------
    signal :  list, array or Series
        The signal channel.
    desired_length : int
        The desired length of the signal.
    sampling_rate, desired_sampling_rate : int
        The original and desired (output) sampling frequency (in Hz, i.e., samples/second).
    method : str
        Can be 'interpolation' (default) or 'FFT' for the Fourier method. FFT is accurate (if the signal is periodic), but slower compared to interpolation.

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
    >>> # Downsample
    >>> signal = np.cos(np.linspace(start=0, stop=20, num=50))
    >>> downsampled_fft = nk.signal_resample(signal, method="FFT", sampling_rate=1000, desired_sampling_rate=500)
    >>> downsampled_interpolation = nk.signal_resample(signal, method="interpolation", sampling_rate=1000, desired_sampling_rate=500)
    >>>
    >>> # Upsample
    >>> upsampled_fft = nk.signal_resample(downsampled_fft, method="FFT", sampling_rate=500, desired_sampling_rate=1000)
    >>> upsampled_interpolation = nk.signal_resample(downsampled_interpolation, method="interpolation", sampling_rate=500, desired_sampling_rate=1000)
    >>>
    >>> # Check
    >>> pd.DataFrame({"Original": signal,
                      "FFT": upsampled_fft,
                      "Interpolation": upsampled_interpolation}).plot(style='.-')
    """
    if desired_length is None:
        desired_length = round(len(signal) * desired_sampling_rate / sampling_rate)

    # Sanity checks
    if len(signal) == desired_length:
        return(signal)

    # Resample
    if method == "FFT":
        resampled = _resample_interpolation(signal, desired_length)
    else:
       resampled =  _resample_fft(signal, desired_length)

    return(resampled)




# =============================================================================
# Internals
# =============================================================================

def _resample_interpolation(signal, desired_length):
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, desired_length, endpoint=False),  # where to interpolate
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return(resampled_signal)




def _resample_fft(signal, desired_length):
    resampled_signal = scipy.signal.resample(signal, desired_length)
    return(resampled_signal)