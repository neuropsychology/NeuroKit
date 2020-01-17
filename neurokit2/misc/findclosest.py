import numpy as np


def findclosest(number, numbers, direction="both", strictly=False):
    """
    Find the closest number in the array from a given number x.

    Parameters
    ----------
    number : float
        The number.
    numbers : list
        The list to look in.
    direction : str
        "both" for smaller or greater, "greater" for only greater numbers and "smaller" for the closest smaller.
    strictly : bool
        False for stricly superior or inferior or True for including equal.

    Returns
    ----------
    closest : int
        The closest number in the array.

    Example
    ----------
    >>> import neurokit2 as nk
    >>> nk.findclosest(1.8, [3, 5, 6, 1, 2])
    """
    if direction == "both":
        closest = min(numbers, key=lambda x:np.abs(x-number))
    if direction == "smaller":
        if strictly is True:
            closest = max(x for x in numbers if x < number)
        else:
            closest = max(x for x in numbers if x <= number)
    if direction == "greater":
        if strictly is True:
            closest = min(filter(lambda x: x > number, numbers))
        else:
            closest = min(filter(lambda x: x >= number, numbers))

    return closest