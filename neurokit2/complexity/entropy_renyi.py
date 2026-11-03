import numpy as np

from .entropy_shannon import _entropy_freq


def entropy_renyi(signal=None, alpha=1, symbolize=None, show=False, freq=None, **kwargs):
    """**Rényi entropy (REn or H)**

    In information theory, the Rényi entropy *H* generalizes the Hartley entropy, the Shannon
    entropy, the collision entropy and the min-entropy.

    * :math:`\\alpha = 0`: the Rényi entropy becomes what is known as the **Hartley entropy**, the
      logarithm of the number of states that occur.
    * :math:`\\alpha = 1`: the Rényi entropy becomes the :func:`Shannon entropy <entropy_shannon>`.
    * :math:`\\alpha = 2`: the Rényi entropy becomes the **collision entropy**, which corresponds
      to the surprisal of "rolling doubles".
    * :math:`\\alpha = \\infty` (``np.inf``): the Rényi entropy becomes the **min-entropy**,
      :math:`-\\log(\\max_{x} p(x))`.

    It is mathematically defined as:

    .. math::

      REn = \\frac{1}{1-\\alpha} \\log \\left( \\sum_{x \\in \\mathcal{X}} p(x)^\\alpha \\right)

    It uses the natural logarithm, giving a unit in *nats* (so that ``alpha=1`` is the same as
    ``entropy_shannon(base=np.e)``). States that do not occur (with a probability of 0) do not
    contribute.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series, str]
        The signal (i.e., a time series) in the form of a vector of values. A string (e.g.,
        ``"ABBA"``) is treated as a sequence of characters.
    alpha : float
        The *alpha* :math:`\\alpha` parameter (default to 1) for Rényi entropy.
    symbolize : str
        Method to convert a continuous signal input into a symbolic (discrete) signal. ``None`` by
        default, which skips the process (and assumes the input is already discrete). See
        :func:`complexity_symbolize` for details.
    show : bool
        If ``True``, will show the discrete signal.
    freq : np.array
        Instead of a signal, a vector of probabilities (or counts, which are normalized) of each
        state can be provided.
    **kwargs
        Other arguments passed to :func:`complexity_symbolize`, such as ``c`` (the number of
        symbols for the methods that need it) or ``random_state``.

    Returns
    --------
    ren : float
        The Rényi entropy of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_shannon, entropy_tsallis

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = [1, 3, 3, 2, 6, 6, 6, 1, 0]
      ren, _ = nk.entropy_renyi(signal, alpha=1)
      ren

      # Compare to Shannon function
      shanen, _ = nk.entropy_shannon(signal, base=np.e)
      shanen

      # Hartley Entropy
      nk.entropy_renyi(signal, alpha=0)[0]

      # Collision Entropy
      nk.entropy_renyi(signal, alpha=2)[0]

      # Min-entropy
      nk.entropy_renyi(signal, alpha=np.inf)[0]


    References
    -----------
    * Rényi, A. (1961, January). On measures of entropy and information. In Proceedings of the
      Fourth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1:
      Contributions to the Theory of Statistics (Vol. 4, pp. 547-562). University of California
      Press.

    """
    if freq is None:
        _, freq = _entropy_freq(signal, symbolize=symbolize, show=show, **kwargs)
    freq = np.asarray(freq, dtype=float)
    freq = freq / np.sum(freq)

    # States that do not occur do not contribute (0 * log(0) = 0, and 0^0 would count them)
    freq = freq[freq != 0]

    if np.isclose(alpha, 1):
        ren = -np.sum(freq * np.log(freq))
    elif alpha == np.inf:
        ren = -np.log(np.max(freq))
    else:
        ren = (1 / (1 - alpha)) * np.log(np.sum(freq**alpha))

    return ren, {"Symbolization": symbolize, "Alpha": alpha}
