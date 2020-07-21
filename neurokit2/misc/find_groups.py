import itertools


def find_groups(x):
    """Find and group repeating (identical) values in a list.

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
    >>> x = [2, 2, 2, 2, 1, 3, 3, 2, 2, 2, 1]
    >>> groups = nk.find_groups(x)
    >>> groups #doctest: +ELLIPSIS
    [[2, 2, 2, 2], [1], [3, 3], [2, 2, 2], [1]]

    """

    return [list(j) for i, j in itertools.groupby(x)]
