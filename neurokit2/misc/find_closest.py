import numpy as np

from .sanitize_input import sanitize_input


def find_closest(closest_of, list_to_search_in, direction="both", strictly=False, return_index=False):
    """
    Find the closest number in the array from a given number x.

    Parameters
    ----------
    closest_of : float
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
    >>> nk.find_closest(1.8, [3, 5, 6, 1, 2])
    >>> nk.find_closest(1.8, [3, 5, 6, 1, 2], return_index=True)
    >>>
    >>> # Vectorized version
    >>> nk.find_closest([1.8, 3.6], [3, 5, 6, 1, 2])
    """

    # Transform to arrays
    closest_of = sanitize_input(closest_of, "vector")
    list_to_search_in = sanitize_input(list_to_search_in, "vector")

    out = [_find_closest(list_to_search_in, i, direction, strictly, return_index) for i in closest_of]

    if len(out) == 1:
        return out[0]
    else:
        return out



# =============================================================================
# Internal
# =============================================================================
def _find_closest(closest_of, list_to_search_in, direction="both", strictly=False, return_index=False):

    try:
        index, closest = _find_closest_single_numpy(list_to_search_in, closest_of, direction, strictly)
    except ValueError:
        index, closest = np.nan, np.nan

    if return_index is True:
        return index
    else:
        return closest


# =============================================================================
# Methods
# =============================================================================

def _find_closest_single_numpy(x, vals, direction="both", strictly=False):


    if direction == "both":
        index = (np.abs(vals - x)).argmin()
        closest = vals[index]

    if direction == "smaller":
        if strictly is True:
            index = (np.abs(vals[vals < x] - x)).argmin()
            closest = vals[vals < x][index]
        else:
            index = (np.abs(vals[vals <= x] - x)).argmin()
            closest = vals[vals <= x][index]

    if direction == "greater":
        if strictly is True:
            index = (np.abs(vals[vals > x] - x)).argmin()
            closest = vals[vals > x][index]
        else:
            index = (np.abs(vals[vals >= x] - x)).argmin()
            closest = vals[vals >= x][index]

    return index, closest
