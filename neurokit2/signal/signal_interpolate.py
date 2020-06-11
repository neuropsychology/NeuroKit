# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate


def signal_interpolate(x_values, y_values, new_x=None, method="quadratic"):
    """Interpolate a signal.

    Interpolate a signal using different methods.

    Parameters
    ----------
    x_values : list, array or Series
        The samples corresponding to the values to be interpolated.
    y_values : list, array or Series
        The values to be interpolated.
    new_x : list, array Series or int
        The samples at which to interpolate the y_values. Samples before the first value in x_values
        or after the last value in x_values will be extrapolated.
        If an integer is passed, nex_x will be considered as the desired length of the interpolated
        signal between the first and the last values of x_values. No extrapolation will be done for values
        before or after the first and the last valus of x_values.
    method : str
        Method of interpolation. Can be 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous', 'next' or 'monotone_cubic'.  'zero', 'slinear', 'quadratic' and 'cubic' refer to
        a spline interpolation of zeroth, first, second or third order; 'previous' and 'next' simply
        return the previous or next value of the point) or as an integer specifying the order of the
        spline interpolator to use.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
        for details on the 'monotone_cubic' method.

    Returns
    -------
    array
        Vector of interpolated samples.

    Examples
    --------
    >>> import numpy as np
    >>> import neurokit2 as nk
    >>> import matplotlib.pyplot as plt
    >>>
    >>> samples = np.linspace(start=0, stop=20, num=10)
    >>> signal = np.cos(samples)
    >>> interpolation_methods = ["zero", "linear", "quadratic", "cubic", "nearest", "monotone_cubic"]
    >>>
    >>> fig, ax = plt.subplots() #doctest: +SKIP
    >>> ax.scatter(samples, signal, label="original datapoints", zorder=3) #doctest: +SKIP
    >>> for im in interpolation_methods:
    ...     signal_interpolated = nk.signal_interpolate(samples, signal, x_new=np.arange(1000), method=im)
    ...     ax.plot(np.linspace(0, 20, 1000), signal_interpolated, label=im) #doctest: +SKIP
    >>> ax.legend(loc="upper left") #doctest: +SKIP

    """
    # Sanity checks
    if len(x_values) != len(y_values):
        raise ValueError("NeuroKit error: signal_interpolate(): x_values and y_values must be of the same length.")

    if isinstance(new_x, int):
        if len(x_values) == new_x:
            return y_values
    else:
        if len(x_values) == len(new_x):
            return y_values

    # Create interpolation function
    if method.lower() != "monotone_cubic":
        interpolation_function = scipy.interpolate.interp1d(
            x_values, y_values, kind=method, bounds_error=False, fill_value=([y_values[0]], [y_values[-1]])
        )
    elif method.lower() == "monotone_cubic":
        interpolation_function = scipy.interpolate.PchipInterpolator(x_values, y_values, extrapolate=True)

    if isinstance(new_x, int):
        new_x = np.linspace(x_values[0], x_values[-1], new_x)

    interpolated = interpolation_function(new_x)

    return interpolated
