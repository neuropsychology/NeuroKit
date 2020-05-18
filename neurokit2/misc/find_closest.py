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

    out = [_find_closest(i, list_to_search_in, direction, strictly, return_index) for i in closest_of]

    if len(out) == 1:
        return out[0]
    else:
        return np.array(out)



# =============================================================================
# Internal
# =============================================================================
def _find_closest(closest_of, list_to_search_in, direction="both", strictly=False, return_index=False):

#    try:
#        index, closest = _find_closest_single_numpy(closest_of, list_to_search_in, direction, strictly)
#    except ValueError:
#        index, closest = np.nan, np.nan


#    if return_index is True:
#        return index
#    else:
#        return closest

    try:
        closest = _findclosest_base(closest_of, list_to_search_in, direction, strictly)
    except ValueError:
        return np.nan

    if return_index is True:
        closest = np.where(np.asarray(list_to_search_in) == closest)[0]
        if len(closest) == 1:
            closest = closest[0]
    return closest


# =============================================================================
# Methods
# =============================================================================

def _findclosest_base(target_number, vals, direction="both", strictly=False):
    if direction == "both":
        closest = min(vals, key=lambda x: np.abs(x-target_number))
    if direction == "smaller":
        if strictly is True:
            closest = max(x for x in vals if x < target_number)
        else:
            closest = max(x for x in vals if x <= target_number)
    if direction == "greater":
        if strictly is True:
            closest = min(filter(lambda x: x > target_number, vals))
        else:
            closest = min(filter(lambda x: x >= target_number, vals))

    return closest





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
            index = (vals[vals > x] - x).argmin()
            closest = vals[vals > x][index]
        else:
            index = (vals[vals >= x] - x).argmin()
            closest = vals[vals >= x][index]

    return index, closest
