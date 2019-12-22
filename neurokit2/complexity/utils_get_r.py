# -*- coding: utf-8 -*-
import numpy as np


def _get_r(signal, r="default"):

    if isinstance(r, str):
        r = 0.2 * np.std(signal, ddof=1)

    return r
