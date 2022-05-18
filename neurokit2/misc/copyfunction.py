import functools


def copyfunction(func, *args, **kwargs):
    """**Copy Function**
    """

    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func
