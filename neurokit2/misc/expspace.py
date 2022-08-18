import numpy as np


def expspace(start, stop, num=50, out=int, base=1):
    """**Exponential range**

    Creates a list of integer values (by default) of a given length from start to stop, spread by
    an exponential function.

    Parameters
    ----------
    start : int
        Minimum range values.
    stop : int
        Maximum range values.
    num : int
        Number of samples to generate. Default is 50. Must be non-negative.
    out : type
        Type of the returned values. Default is int.
    base : float
        If 1, will use :func:`.np.exp`, if 2 will use :func:`.np.exp2`.

    Returns
    -------
    array
        An array of integer values spread by the exponential function.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk
      nk.expspace(start=4, stop=100, num=10)
      nk.expspace(start=0, stop=1, num=10, out=float)

    """
    if base == 1:
        seq = np.exp(np.linspace(np.log(1 + start), np.log(1 + stop), num, endpoint=True)) - 1
    else:
        seq = np.exp2(
            np.linspace(np.log2(1 + start), np.log2(1 + stop), num, endpoint=True) - 1
        )  # pylint: disable=E1111

    # Round and convert to int
    if out == int:
        seq = np.round(seq).astype(int)

    return seq
