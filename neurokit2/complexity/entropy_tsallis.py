import numpy as np

from .entropy_shannon import _entropy_freq


def entropy_tsallis(signal=None, q=1, symbolize=None, show=False, freq=None, **kwargs):
    """**Tsallis entropy (TSEn)**

    Tsallis Entropy is an extension of :func:`Shannon entropy <entropy_shannon>` to the case where
    entropy is nonextensive. It is similarly computed from a vector of probabilities of different
    states. Because it works on discrete inputs (e.g., [A, B, B, A, B]), it requires to transform
    the continuous signal into a discrete one.


    .. math::

      TSEn = \\frac{1}{q - 1} \\left( 1 - \\sum_{x \\in \\mathcal{X}} p(x)^q \\right)


    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    q : float
        Tsallis's *q* parameter, sometimes referred to as the entropic-index (default to 1).
    symbolize : str
        Method to convert a continuous signal input into a symbolic (discrete) signal. ``None`` by
        default, which skips the process (and assumes the input is already discrete). See
        :func:`complexity_symbolize` for details.
    show : bool
        If ``True``, will show the discrete the signal.
    freq : np.array
        Instead of a signal, a vector of probabilities can be provided.
    **kwargs
        Optional arguments. Not used for now.

    Returns
    --------
    tsen : float
        The Tsallis entropy of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_shannon, fractal_petrosian, entropy_renyi

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = [1, 3, 3, 2, 6, 6, 6, 1, 0]
      tsen, _ = nk.entropy_tsallis(signal, q=1)
      tsen

      shanen, _ = nk.entropy_shannon(signal, base=np.e)
      shanen


    References
    -----------
    * Tsallis, C. (2009). Introduction to nonextensive statistical mechanics: approaching a complex
      world. Springer, 1(1), 2-1.

    """
    if freq is None:
        _, freq = _entropy_freq(signal, symbolize=symbolize, show=show)
    freq = freq / np.sum(freq)

    if np.isclose(q, 1):
        lnq_1_over_p = np.log(1 / freq)
    else:
        lnq_1_over_p = ((1 / freq) ** (1 - q) - 1) / (1 - q)

    tsens = freq * lnq_1_over_p
    return np.sum(tsens), {"Symbolization": symbolize, "Values": tsens}
