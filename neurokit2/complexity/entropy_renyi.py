import numpy as np

from .entropy_shannon import _entropy_freq


def entropy_renyi(signal, alpha=1, method=None, show=False, freq=None):
    """**Rényi entropy (REn)**

    In information theory, the Rényi entropy generalizes the Hartley entropy, the Shannon entropy,
    the collision entropy and the min-entropy.

    - :math:`\\alpha = 0`: the Rényi entropy becomes what is known as the **Hartley entropy**.
    - :math:`\\alpha = 1`: the Rényi entropy becomes the :func:`**Shannon entropy** <entropy_shannon>`.
    - :math:`\\alpha = 2`: the Rényi entropy becomes the collision entropy, which corresponds to
      the surprisal of "rolling doubles".

    It is mathematically defined as:

     .. math::

    REn = \\frac{1}{1-\\alpha} \\log_2 \\left( \\sum_{x in \\mathcal{X}} p(x)^\\alpha \\right)

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    alpha : float
        Rényi's *alpha* parameter (default to 1).
    method : str or int
        Method of discretization. Can be one of ``"A"``, ``"B"``, ``"C"``, ``"D"``, ``"r"``, an
        ``int`` indicating the number of bins, or ``None`` to skip the process (for instance, in
        cases when the binarization has already been done before). See :func:`fractal_petrosian`
        for details.
    show : bool
        If ``True``, will show the discrete the signal.

    Returns
    --------
    ren : float
        The Tsallis entropy of the signal.
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
      tsen, _ = nk.entropy_renyi(signal, alpha=1)
      tsen

      shanen, _ = nk.entropy_shannon(signal, base=np.e)
      shanen

      # Hartley Entropy
      nk.entropy_renyi(signal, alpha=0)[0]

      # Collision Entropy
      nk.entropy_renyi(signal, alpha=2)[0]



    References
    -----------
    * Rényi, A. (1961, January). On measures of entropy and information. In Proceedings of the
      Fourth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1:
      Contributions to the Theory of Statistics (Vol. 4, pp. 547-562). University of California
      Press.

    """
    if freq is None:
        freq = _entropy_freq(signal, method=method, show=show)

    freq = freq / np.sum(freq)
    if np.isclose(alpha, 1):
        ren = -np.sum(freq * np.log(freq))
    else:
        ren = (1 / (1 - alpha)) * np.log(np.sum(freq ** alpha))

    return ren, {"Method": method}
