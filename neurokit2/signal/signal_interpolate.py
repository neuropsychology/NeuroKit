# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate


def signal_interpolate(signal, desired_length=None, method="quadratic",
                       x_axis=None, new_x=None):
    """Interpolate a signal.

    Samples up until the first peak as well as from last peak to end of signal
    are set to the value of the first and last element of 'stats' respectively.
    Linear (2nd order) interpolation is chosen since higher order interpolation
    can lead to biologically implausible values and erratic fluctuations.

    Parameters
    ----------
    signal : list, array or Series
        The signal channel in the form of a vector of values.
    desired_length : int
        The desired length of the signal.
    method : str
        Method of interpolation. Can be 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous' or 'next'.  'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of zeroth, first, second or third order; 'previous' and 'next' simply return the previous or next value of the point) or as an integer specifying the order of the spline interpolator to use.
    x_axis : list, array or Series
        An optional vector of same length as 'signal' corresponding to the x-axis.

    Returns
    -------
    array
        Vector of interpolated samples.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>> import matplotlib.pyplot as plt
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=20, num=10))
    >>> zero = nk.signal_interpolate(signal, desired_length=1000, method="zero")
    >>> linear = nk.signal_interpolate(signal, desired_length=1000, method="linear")
    >>> quadratic = nk.signal_interpolate(signal, desired_length=1000, method="quadratic")
    >>> cubic = nk.signal_interpolate(signal, desired_length=1000, method="cubic")
    >>> nearest = nk.signal_interpolate(signal, desired_length=1000, method="nearest")
    >>>
    >>> plt.plot(np.linspace(0, 1, num=len(zero)), zero, 'y',
                 np.linspace(0, 1, num=len(linear)), linear, 'r',
                 np.linspace(0, 1, num=len(quadratic)), quadratic, 'b',
                 np.linspace(0, 1, num=len(cubic)), cubic, 'g',
                 np.linspace(0, 1, num=len(nearest)), nearest, 'm',
                 np.linspace(0, 1, num=len(signal)), signal, 'ko')
    >>>
    >>> # Use x-axis end new x-axis
    >>> x_axis = np.linspace(start=10, stop=30, num=10)
    >>> signal = np.cos(x_axis)
    >>> new_x = np.linspace(start=0, stop=40, num=1000)
    >>> interpolated = nk.signal_interpolate(signal,
                                    desired_length=1000,
                                    x_axis=x_axis,
                                    new_x=new_x)
    >>> plt.plot(new_x, interpolated, '-',
                 x, signal, 'o')
    """
    # Sanity checks
    if desired_length is None:
        if new_x is not None:
            desired_length = new_x
        else:
            raise ValueError("NeuroKit error: signal_interpolate(): either 'desired_length' or 'new_x' must be provided.")
    if desired_length < len(signal):
        raise ValueError("NeuroKit error: signal_interpolate(): 'desired_length' cannot be lower than the length of the signal. You might be looking for 'signal_resample()'")

    # Create x axis
    if x_axis is None:
        x_axis = np.arange(0, len(signal))

    # Create interpolation function
    interpolation_function = scipy.interpolate.interp1d(
            x_axis,
            np.ravel(signal),
            kind=method,
            bounds_error=False,
            fill_value=([signal[0]], [signal[-1]]))

    if new_x is None:
        new_x = np.linspace(x_axis[0], x_axis[-1], num=desired_length)

    interpolated = interpolation_function(new_x)

    return(interpolated)
