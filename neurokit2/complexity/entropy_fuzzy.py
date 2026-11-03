from .entropy_approximate import entropy_approximate
from .entropy_sample import entropy_sample


def entropy_fuzzy(signal, delay=1, dimension=2, tolerance="sd", approximate=False, **kwargs):
    """**Fuzzy Entropy (FuzzyEn)**

    Fuzzy entropy (FuzzyEn) of a signal stems from the combination between information theory and
    fuzzy set theory (Zadeh, 1965). A fuzzy set is a set containing elements with varying degrees of
    membership.

    FuzzyEn (Chen et al., 2007) follows the steps of :func:`SampEn <entropy_sample>`, with two
    differences: the baseline (mean) of each embedded vector is removed, and the similarity of two
    vectors is not 1 or 0 depending on whether their Chebyshev distance *d* is within the tolerance
    *r*, but given by a fuzzy membership function, :math:`\\mu(d) = e^{-d^n / r}` (by default
    with :math:`n = 1`, while Chen et al., 2007, used :math:`n = 2`). For each of the
    :math:`N - m \\tau` templates, the similarities to the other templates (self-matches excluded)
    are averaged, and :math:`\\phi^m` is the mean of these averages, so that

    .. math::

      FuzzyEn = \\ln(\\phi^m) - \\ln(\\phi^{m+1})

    The approximate version (FuzzyApEn) follows instead the steps of :func:`ApEn
    <entropy_approximate>`, with the same baseline removal and membership function: it is
    :math:`\\Phi^m - \\Phi^{m+1}`, where :math:`\\Phi^m` is the average of the logarithm of the
    mean similarity of each vector to all the vectors, itself included (as in ApEn, unlike the
    fuzzy approximate entropy of Xie et al., 2010, which excludes it).

    This function can be called either via ``entropy_fuzzy()`` or ``complexity_fuzzyen()``, or
    ``complexity_fuzzyapen()`` for its approximate version. Note that the fuzzy corrected
    approximate entropy (cApEn) can also be computed via setting ``corrected=True`` along with
    ``approximate=True`` (see examples).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    tolerance : float
        Tolerance (often denoted as *r*), the width of the fuzzy membership function. If
        ``"sd"`` (default), will be set to :math:`0.2 * SD_{signal}`. See
        :func:`complexity_tolerance` to estimate the optimal value for this parameter.
    approximate : bool
        If ``True``, will compute the fuzzy approximate entropy (FuzzyApEn).
    **kwargs
        Other arguments, such as the exponent ``n`` of the membership function (1 by default,
        while Chen et al., 2007, used ``n=2``), or ``corrected=True`` (with ``approximate=True``)
        for the corrected version of FuzzyApEn (see :func:`entropy_approximate`).

    Returns
    ----------
    fuzzyen : float
        The fuzzy entropy of the single time series.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute fuzzy entropy.

    See Also
    --------
    entropy_sample, entropy_approximate, entropy_multiscale

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=5)

      fuzzyen, parameters = nk.entropy_fuzzy(signal)
      fuzzyen

      fuzzyapen, parameters = nk.entropy_fuzzy(signal, approximate=True)
      fuzzyapen

      fuzzycapen, parameters = nk.entropy_fuzzy(signal, approximate=True, corrected=True)
      fuzzycapen


    References
    ----------
    * Chen, W., Wang, Z., Xie, H., & Yu, W. (2007). Characterization of surface EMG signal based on
      fuzzy entropy. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 15(2),
      266-272.
    * Chen, W., Zhuang, J., Yu, W., & Wang, Z. (2009). Measuring complexity using FuzzyEn, ApEn,
      and SampEn. Medical Engineering & Physics, 31(1), 61-68.
    * Xie, H. B., Guo, J. Y., & Zheng, Y. P. (2010). Fuzzy approximate entropy analysis of chaotic
      and natural complex systems: detecting muscle fatigue using electromyography signals. Annals
      of Biomedical Engineering, 38(4), 1483-1496.
    * Ishikawa, A., & Mieno, H. (1979). The fuzzy entropy concept and its application. Fuzzy Sets
      and systems, 2(2), 113-123.
    * Zadeh, L. A. (1965). Fuzzy sets. Information and Control, 8(3), 338-353.

    """
    if approximate is False:
        if kwargs.get("corrected", False) is True:
            raise ValueError("The corrected version of FuzzyEn is only available with `approximate=True`.")
        out = entropy_sample(
            signal,
            delay=delay,
            dimension=dimension,
            tolerance=tolerance,
            fuzzy=True,
            **kwargs,
        )
    else:
        out = entropy_approximate(
            signal,
            delay=delay,
            dimension=dimension,
            tolerance=tolerance,
            fuzzy=True,
            **kwargs,
        )
    return out
