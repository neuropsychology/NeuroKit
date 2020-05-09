import numpy as np


def range_log(start, stop, factor=1):
    """Logarithmic range

    Creates a list of integer values without duplicates by successively multiplying a minimum
    value min_n by a factor > 1 until a maximum value max_n is reached.

    Non-integer results are rounded down. This is the vectorized version of ``nolds.logarithmic_n()``.

    Parameters
    ----------
    start, stop : int
        Minimum and maximum range values.
    factor (float):
        Logarithmic factor used to increase min_n (must be > 1)

    Examples
    ---------
    >>> import neurokit2 as nk
    >>> nk.range_log(start=4, stop=20, factor=1.1)
    """
    end_i = np.int(np.floor(np.log(1.0 * stop / start) / np.log(factor)))

    windows = np.arange(end_i + 1)
    windows = np.floor(start * (factor ** windows))
    windows = np.unique(windows)

    return windows.astype(int)
