# -*- coding: utf-8 -*-
import numpy as np


def rescale(data, to=[0, 1], scale=None):
    """**Rescale data**

    Rescale a numeric variable to a new range.

    Parameters
    ----------
    data : Union[list, np.array, pd.Series]
        Raw data.
    to : list
        New range of values of the data after rescaling.
    scale : list
        A list or tuple of two values specifying the actual range
        of the data. If ``None``, the minimum and the maximum of the
        provided data will be used.

    Returns
    ----------
    list
        The rescaled values.


    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      nk.rescale([3, 1, 2, 4, 6], to=[0, 1])

    """

    # Return appropriate type
    if isinstance(data, list):
        data = list(_rescale(np.array(data), to=to, scale=scale))
    else:
        data = _rescale(data, to=to, scale=scale)

    return data


# =============================================================================
# Internals
# =============================================================================
def _rescale(data, to=[0, 1], scale=None):
    if scale is None:
        scale = [np.nanmin(data), np.nanmax(data)]

    return (to[1] - to[0]) / (scale[1] - scale[0]) * (data - scale[0]) + to[0]
