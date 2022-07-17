# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def fractal_linelength(signal):
    """**Line Length (LL)**

    Line Length (LL, also known as curve length), stems from a modification of the
    :func:`Katz fractal dimension <fractal_katz>` algorithm, with the goal of making it more
    efficient and accurate (especially for seizure onset detection).

    It basically corresponds to the average of the absolute consecutive differences of the signal,
    and was made to be used within subwindows. Note that this does not technically measure the
    fractal dimension, but the function was named with the ``fractal_`` prefix due to its
    conceptual similarity with Katz's fractal dimension.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    Returns
    -------
    float
        Line Length.
    dict
        A dictionary containing additional information (currently empty, but returned nonetheless
        for consistency with other functions).

    See Also
    --------
    fractal_katz

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6, 10])

      ll, _ = nk.fractal_linelength(signal)
      ll


    References
    ----------
    * Esteller, R., Echauz, J., Tcheng, T., Litt, B., & Pless, B. (2001, October). Line length: an
      efficient feature for seizure onset detection. In 2001 Conference Proceedings of the 23rd
      Annual International Conference of the IEEE Engineering in Medicine and Biology Society (Vol.
      2, pp. 1707-1710). IEEE.

    """

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Force to array
    signal = np.array(signal)

    # Drop missing values
    signal = signal[~np.isnan(signal)]

    # Compute line length
    ll = np.mean(np.abs(np.diff(signal)))

    return ll, {}
