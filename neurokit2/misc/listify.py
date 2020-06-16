# -*- coding: utf-8 -*-


def listify(**kwargs):
    """Transforms arguments into lists of the same length.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> nk.listify(a=3, b=[3, 5], c=[3]) #doctest: +ELLIPSIS
    {'a': [3, 3], 'b': [3, 5], 'c': [3, 3]}

    """
    args = kwargs
    maxi = 1

    # Find max length
    for key, value in args.items():
        if isinstance(value, str) is False:
            try:
                if len(value) > maxi:
                    maxi = len(value)
            except TypeError:
                pass

    # Transform to lists
    for key, value in args.items():
        if isinstance(value, list):
            args[key] = _multiply_list(value, maxi)
        else:
            args[key] = _multiply_list([value], maxi)

    return args


def _multiply_list(lst, length):
    q, r = divmod(length, len(lst))
    return q * lst + lst[:r]
