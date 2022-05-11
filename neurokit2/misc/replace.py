# -*- coding: utf-8 -*-
import numpy as np


def replace(data, replacement_dict):
    """**Replace values using a dictionary**

    Parameters
    ----------
    data : array
        The data to replace values.
    replacement_dict : dict
        A replacement dictionary of the form ``{old_value: new_value}``.

    Returns
    -------
    array
        Array containing the replaced values.

    Examples
    --------
      import neurokit2 as nk

      data = [0, 1, 2, 3]
      replacement = {0: 99, 3: 42}
      replaced = nk.replace(data, replacement)
      replaced

    """
    # Extract out keys and values
    k = np.array(list(replacement_dict.keys()))
    v = np.array(list(replacement_dict.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    idx = np.searchsorted(ks, data)

    idx[idx == len(vs)] = 0
    mask = ks[idx] == data
    return np.where(mask, vs[idx], data)
