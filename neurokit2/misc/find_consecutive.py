import itertools


def find_consecutive(x):
    """Find and group consecutive values in a list.

    Parameters
    ----------
    x : list
        The list to look in.

    Returns
    -------
    list
        A list of tuples corresponding to groups containing all the consecutive numbers.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> x = [2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 20]
    >>> nk.find_consecutive(x)
    [(2, 3, 4, 5), (12, 13, 14, 15, 16, 17), (20,)]

    """

    return [tuple(g) for k, g in itertools.groupby(x, lambda n, c=itertools.count(): n - next(c))]
