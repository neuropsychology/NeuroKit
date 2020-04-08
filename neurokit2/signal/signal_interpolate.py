# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate

def signal_interpolate(x_values, y_values, desired_length, method="quadratic"):
    """Interpolate a signal.

    Interpolate (fills the values between data points) a signal using different methods.

    Parameters
    ----------
    x_values : list, array or Series
        The samples corresponding to the values to be interpolated.
    y_values : list, array or Series
        The values to be interpolated.
    desired_length : int
        The amount of samples over which to interpolate the y_values.
    method : str
        Method of interpolation. Can be 'linear', 'nearest', 'zero', 'slinear',
        'quadratic', 'cubic', 'previous' or 'next'.  'zero', 'slinear',
        'quadratic' and 'cubic' refer to a spline interpolation of zeroth,
        first, second or third order; 'previous' and 'next' simply return the
        previous or next value of the point) or as an integer specifying the
        order of the spline interpolator to use.

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
    if len(x_values) != len(y_values):
        raise ValueError("NeuroKit error: signal_interpolate(): x_values and y_values "
                         "must be of the same length.")

    if desired_length is None or len(x_values) == desired_length:
        return y_values


    # Create interpolation function
    interpolation_function = scipy.interpolate.interp1d(x_values,
                                                        y_values,
                                                        kind=method,
                                                        bounds_error=False,
                                                        fill_value=([y_values[0]], [y_values[-1]]))

    new_x = np.arange(desired_length)

    interpolated = interpolation_function(new_x)

    return interpolated
