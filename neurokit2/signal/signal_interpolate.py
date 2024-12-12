# -*- coding: utf-8 -*-
from warnings import warn

import numpy as np
import pandas as pd
import scipy.interpolate

from ..misc import NeuroKitWarning


def signal_interpolate(
    x_values, y_values=None, x_new=None, method="quadratic", fill_value=None
):
    """**Interpolate a signal**

    Interpolate a signal using different methods.

    Parameters
    ----------
    x_values : Union[list, np.array, pd.Series]
        The samples corresponding to the values to be interpolated.
    y_values : Union[list, np.array, pd.Series]
        The values to be interpolated. If not provided, any NaNs in the x_values
        will be interpolated with :func:`_signal_interpolate_nan`,
        considering the x_values as equally spaced.
    x_new : Union[list, np.array, pd.Series] or int
        The samples at which to interpolate the y_values. Samples before the first value in x_values
        or after the last value in x_values will be extrapolated. If an integer is passed, nex_x
        will be considered as the desired length of the interpolated signal between the first and
        the last values of x_values. No extrapolation will be done for values before or after the
        first and the last values of x_values.
    method : str
        Method of interpolation. Can be ``"linear"``, ``"nearest"``, ``"zero"``, ``"slinear"``,
        ``"quadratic"``, ``"cubic"``, ``"previous"``, ``"next"``, ``"monotone_cubic"``, or ``"akima"``.
        The methods ``"zero"``, ``"slinear"``, ``"quadratic"`` and ``"cubic"`` refer to a spline
        interpolation of zeroth, first, second or third order; whereas ``"previous"`` and
        ``"next"`` simply return the previous or next value of the point. An integer specifying the
        order of the spline interpolator to use.
        See `monotone cubic method <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.
        PchipInterpolator.html>`_ for details on the ``"monotone_cubic"`` method.
        See `akima method <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.
        Akima1DInterpolator.html>`_ for details on the ``"akima"`` method.
    fill_value : float or tuple or str
        If a ndarray (or float), this value will be used to fill in for
        requested points outside of the data range.
        If a two-element tuple, then the first element is used as a fill value
        for x_new < x[0] and the second element is used for x_new > x[-1].
        If "extrapolate", then points outside the data range will be extrapolated.
        If not provided, then the default is ([y_values[0]], [y_values[-1]]).

    Returns
    -------
    array
        Vector of interpolated samples.

    See Also
    --------
    signal_resample

    Examples
    --------
    .. ipython:: python

      import numpy as np
      import neurokit2 as nk
      import matplotlib.pyplot as plt

      # Generate Simulated Signal
      signal = nk.signal_simulate(duration=2, sampling_rate=10)
      # We want to interpolate to 2000 samples
      x_values = np.linspace(0, 2000, num=len(signal), endpoint=False)
      x_new = np.linspace(0, 2000, num=2000, endpoint=False)

      # Visualize all interpolation methods
      @savefig p_signal_interpolate1.png scale=100%
      nk.signal_plot([
          nk.signal_interpolate(x_values, signal, x_new=x_new, method="zero"),
          nk.signal_interpolate(x_values, signal, x_new=x_new, method="linear"),
          nk.signal_interpolate(x_values, signal, x_new=x_new, method="quadratic"),
          nk.signal_interpolate(x_values, signal, x_new=x_new, method="cubic"),
          nk.signal_interpolate(x_values, signal, x_new=x_new, method="previous"),
          nk.signal_interpolate(x_values, signal, x_new=x_new, method="next"),
          nk.signal_interpolate(x_values, signal, x_new=x_new, method="monotone_cubic")
      ], labels = ["Zero", "Linear", "Quadratic", "Cubic", "Previous", "Next", "Monotone Cubic"])

      # Add original data points
      plt.scatter(x_values, signal, label="original datapoints", zorder=3)
      @suppress
      plt.close()

    """
    # Sanity checks
    if x_values is None:
        raise ValueError(
            "NeuroKit error: signal_interpolate(): x_values must be provided."
        )
    if y_values is None:
        # for interpolating NaNs
        return _signal_interpolate_nan(x_values, method=method, fill_value=fill_value)
    if isinstance(x_values, pd.Series):
        x_values = np.squeeze(x_values.values)
    if isinstance(x_new, pd.Series):
        x_new = np.squeeze(x_new.values)
    if isinstance(y_values, pd.Series):
        y_values = np.squeeze(y_values.values)

    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must be of the same length.")

    if isinstance(x_new, int):
        if len(x_values) == x_new:
            return y_values
        x_new = np.linspace(x_values[0], x_values[-1], x_new)
    else:
        # if x_values is identical to x_new, no need for interpolation
        if np.array_equal(x_values, x_new):
            return y_values
        elif np.any(x_values[1:] == x_values[:-1]):
            warn(
                "Duplicate x values detected. Averaging their corresponding y values.",
                category=NeuroKitWarning,
            )
            x_values, y_values = _signal_interpolate_average_duplicates(
                x_values, y_values
            )

    # If only one value, return a constant signal
    if len(x_values) == 1:
        return np.ones(len(x_new)) * y_values[0]

    if method == "monotone_cubic":
        interpolation_function = scipy.interpolate.PchipInterpolator(
            x_values, y_values, extrapolate=True
        )
    elif method == "akima":
        interpolation_function = scipy.interpolate.Akima1DInterpolator(
            x_values, y_values
        )
    else:
        if fill_value is None:
            fill_value = ([y_values[0]], [y_values[-1]])
        interpolation_function = scipy.interpolate.interp1d(
            x_values,
            y_values,
            kind=method,
            bounds_error=False,
            fill_value=fill_value,
        )

    interpolated = interpolation_function(x_new)

    if method == "monotone_cubic" and fill_value != "extrapolate":
        # Find the index of the new x value that is closest to the first original x value
        first_index = np.argmin(np.abs(x_new - x_values[0]))
        # Find the index of the new x value that is closest to the last original x value
        last_index = np.argmin(np.abs(x_new - x_values[-1]))

        if fill_value is None:
            # Swap out the cubic extrapolation of out-of-bounds segments generated by
            # scipy.interpolate.PchipInterpolator for constant extrapolation akin to the behavior of
            # scipy.interpolate.interp1d with fill_value=([y_values[0]], [y_values[-1]].
            fill_value = ([interpolated[first_index]], [interpolated[last_index]])
        elif isinstance(fill_value, (float, int)):
            # if only a single integer or float is provided as a fill value, format as a tuple
            fill_value = ([fill_value], [fill_value])

        interpolated[:first_index] = fill_value[0]
        interpolated[last_index + 1 :] = fill_value[1]

    return interpolated


def _signal_interpolate_nan(values, method="quadratic", fill_value=None):
    if np.any(np.isnan(values)):
        # assume that values are evenly spaced
        # x_new corresponds to the indices of all values, including missing
        x_new = np.arange(len(values))
        not_missing = np.where(np.invert(np.isnan(values)))[0]

        # remove the missing values
        y_values = values[not_missing]

        # x_values corresponds to the indices of only non-missing values
        x_values = x_new[not_missing]

        # interpolate to get the values at the indices where they are missing
        return signal_interpolate(
            x_values=x_values,
            y_values=y_values,
            x_new=x_new,
            method=method,
            fill_value=fill_value,
        )
    else:
        # if there are no missing values, return original values
        return values


def _signal_interpolate_average_duplicates(x_values, y_values):
    unique_x, indices = np.unique(x_values, return_inverse=True)
    mean_y = np.bincount(indices, weights=y_values) / np.bincount(indices)
    return unique_x, mean_y
