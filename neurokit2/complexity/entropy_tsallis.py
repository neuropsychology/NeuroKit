import numpy as np
import pandas as pd
import scipy.stats

from .entropy_shannon import _entropy_freq


def entropy_tsallis(signal, q=1, method=None, show=False, freq=None):
    """**Tsallis entropy (TSEn)**

    Tsallis Entropy is an extension of :func:`Shannon entropy <entropy_shannon>`, and is similarly
    computed from a vector of probabilities of different states. Because it works on discrete
    inputs (e.g., [A, B, B, A, B]), it requires to transform the continuous signal into a discrete
    one.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    q : float
        Tsallis's *q* parameter, sometimes referred to as the entropic-index (default to 1).
    method : str or int
        Method of discretization. Can be one of ``"A"``, ``"B"``, ``"C"``, ``"D"``, ``"r"``, an
        ``int`` indicating the number of bins, or ``None`` to skip the process (for instance, in
        cases when the binarization has already been done before). See :func:`fractal_petrosian`
        for details.
    show : bool
        If ``True``, will show the discrete the signal.

    Returns
    --------
    shanen : float
        The Shannon entropy of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute Shannon entropy.

    See Also
    --------
    entropy_shannon, fractal_petrosian

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = [1, 3, 3, 2, 6, 6, 6, 1, 0]
      freq = np.unique(signal, return_counts=True)[1]
      tsen, _ = nk.entropy_tsallis(signal, q=1)
      tsen

      shanen, _ = nk.entropy_shannon(signal)
      shanen


    References
    -----------
    * Tsallis, C. (2009). Introduction to nonextensive statistical mechanics: approaching a complex
      world. Springer, 1(1), 2-1.

    """
    if freq is None:
        freq = _entropy_freq(signal, method=method, show=show)

    freq = freq / np.sum(freq)
    if np.isclose(q, 1):
        lnq_1_over_p = np.log(1 / freq)
    else:
        lnq_1_over_p = ((1 / freq) ** (1 - q) - 1) / (1 - q)

    tsens = freq * lnq_1_over_p
    return np.sum(tsens), {"Method": method, "Values": tsens}
