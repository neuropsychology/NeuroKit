import numpy as np
import pandas as pd

from .type_converters import as_vector


def find_closest(closest_to, list_to_search_in, direction="both", strictly=False, return_index=False):
    """
    Find the closest number in the array from a given number x.

    Parameters
    ----------
    closest_to : float
        The target number(s) to find the closest of.
    list_to_search_in : list
        The list of values to look in.
    direction : str
        "both" for smaller or greater, "greater" for only greater numbers and "smaller" for the closest smaller.
    strictly : bool
        False for stricly superior or inferior or True for including equal.
    return_index : bool
        If True, will return the index of the closest value in the list.

    Returns
    ----------
    closest : int
        The closest number in the array.

    Example
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Single number
    >>> x = nk.find_closest(1.8, [3, 5, 6, 1, 2])
    >>> x  #doctest: +SKIP
    >>>
    >>> y = nk.find_closest(1.8, [3, 5, 6, 1, 2], return_index=True)
    >>> y  #doctest: +SKIP
    >>>
    >>> # Vectorized version
    >>> x = nk.find_closest([1.8, 3.6], [3, 5, 6, 1, 2])
    >>> x  #doctest: +SKIP

    """

    # Transform to arrays
    closest_to = as_vector(closest_to)
    list_to_search_in = pd.Series(as_vector(list_to_search_in))

    out = [_find_closest(i, list_to_search_in, direction, strictly, return_index) for i in closest_to]

    if len(out) == 1:
        return out[0]
    else:
        return np.array(out)


# =============================================================================
# Internal
# =============================================================================
def _find_closest(closest_to, list_to_search_in, direction="both", strictly=False, return_index=False):

    try:
        index, closest = _find_closest_single_pandas(closest_to, list_to_search_in, direction, strictly)
    except ValueError:
        index, closest = np.nan, np.nan

    if return_index is True:
        return index
    else:
        return closest


# =============================================================================
# Methods
# =============================================================================


def _findclosest_base(x, vals, direction="both", strictly=False):
    if direction == "both":
        closest = min(vals, key=lambda y: np.abs(y - x))
    if direction == "smaller":
        if strictly is True:
            closest = max(y for y in vals if y < x)
        else:
            closest = max(y for y in vals if y <= x)
    if direction == "greater":
        if strictly is True:
            closest = min(filter(lambda y: y > x, vals))
        else:
            closest = min(filter(lambda y: y >= x, vals))

    return closest


def _find_closest_single_pandas(x, vals, direction="both", strictly=False):

    if direction == "both":
        index = (np.abs(vals - x)).idxmin()

    if direction == "smaller":
        if strictly is True:
            index = (np.abs(vals[vals < x] - x)).idxmin()
        else:
            index = (np.abs(vals[vals <= x] - x)).idxmin()

    if direction == "greater":
        if strictly is True:
            index = (vals[vals > x] - x).idxmin()
        else:
            index = (vals[vals >= x] - x).idxmin()

    closest = vals[index]

    return index, closest
