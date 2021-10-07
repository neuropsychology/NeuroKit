from warnings import warn

import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning


def complexity_k(signal, k_max="default", show=False):
    """
    fractal_higuchi [summary]

    Parameters
    ----------
    signal : [type]
        [description]
    kmax : str, optional
        [description], by default "default"
    show : bool, optional
        [description], by default False

    Reference
    ----------
    - Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory.
    Physica D: Nonlinear Phenomena, 31(2), 277-283.

    - Vega, C. F., & Noel, J. (2015, June). Parameters analyzed of Higuchi's fractal dimension for EEG brain signals.
    In 2015 Signal Processing Symposium (SPSympo) (pp. 1-5). IEEE. https://ieeexplore.ieee.org/document/7168285
    """
    # Get the range of k-max values to be tested
    # ------------------------------------------
    if isinstance(k_max, str):  # e.g., "default"
        # upper limit for k value (max possible value)
        k_max = int(np.floor(len(signal) / 2))  # so that normalizing factor is positive

    if isinstance(k_max, int):
        k_range = np.arange(2, k_max + 1)
    elif isinstance(k_max, [list, np.ndarray, pd.Series]):
        k_range = np.array(k_max)
    else:
        warn(
            "k_max should be an int or a list of values of kmax to be tested.",
            category=NeuroKitWarning,
        )
