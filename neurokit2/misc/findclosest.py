import numpy as np


def findclosest(number, list_of_numbers, direction="both", strictly=False, return_index=False):
    """
    Find the closest number in the array from a given number x.

    Parameters
    ----------
    number : float
        The number.
    list_of_numbers : list
        The list to look in.
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
    >>> nk.findclosest(1.8, [3, 5, 6, 1, 2])
    >>> nk.findclosest(1.8, [3, 5, 6, 1, 2], return_index=True)
    """
    try:
        closest = _findclosest(number, list_of_numbers, direction, strictly)
    except ValueError:
        closest = np.nan

    if return_index is True:
        closest = np.where(np.asarray(list_of_numbers) == closest)[0]
        if len(closest) == 1:
            closest = closest[0]
    return closest




def _findclosest(number, list_of_numbers, direction="both", strictly=False):
    if direction == "both":
        closest = min(list_of_numbers, key=lambda x: np.abs(x-number))
    if direction == "smaller":
        if strictly is True:
            closest = max(x for x in list_of_numbers if x < number)
        else:
            closest = max(x for x in list_of_numbers if x <= number)
    if direction == "greater":
        if strictly is True:
            closest = min(filter(lambda x: x > number, list_of_numbers))
        else:
            closest = min(filter(lambda x: x >= number, list_of_numbers))

    return closest
