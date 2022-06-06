from .entropy_approximate import entropy_approximate
from .entropy_sample import entropy_sample


def entropy_range(signal, dimension=3, delay=1, tolerance="default", approximate=False, **kwargs):
    """**Range Entropy (RangeEn)**

    Introduced by `Omidvarnia et al. (2018) <https://www.mdpi.com/1099-4300/20/12/962/htm>`_,
    RangeEn refers to a modified form of SampEn (or ApEn).

    Both ApEn and SampEn compute the logarithmic likelihood that runs of patterns that are close
    remain close on the next incremental comparisons, of which this closeness is estimated by the
    Chebyshev distance. Range Entropy uses instead a normalized "range distance", resulting in
    modified forms of ApEn and SampEn, **RangeEn (A)** (*mApEn*) and **RangeEn (B)** (*mSampEn*).

    However, the RangeEn (A), based on ApEn, often yields undefined entropies (i.e., *NaN* or
    *Inf*). As such, using RangeEn (B) is recommended instead.

    RangeEn is described as more robust to nonstationary signal changes, and has a more linear
    relationship with the Hurst exponent (compared to ApEn and SampEn), and has no need for signal
    amplitude correction.

    Note that the :func:`corrected <entropy_approximate>` version of ApEn (cApEn) can be computed
    by setting ``corrected=True``.



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
        Tolerance (often denoted as *r*), distance to consider two data points as similar. If
        ``"sd"`` (default), will be set to :math:`0.2 * SD_{signal}`. See
        :func:`complexity_tolerance` to estimate the optimal value for this parameter.
    approximate : bool
        The entropy algorithm to use. If ``False`` (default), will use sample entropy and return
        *mSampEn* (**RangeEn B**). If ``True``, will use approximate entropy and return *mApEn*
        (**RangeEn A**).
    **kwargs
        Other arguments.

    See Also
    --------
    entropy_approximate, entropy_sample

    Returns
    -------
    RangeEn : float
        Range Entropy. If undefined conditional probabilities are detected (logarithm
        of sum of conditional probabilities is ``ln(0)``), ``np.inf`` will
        be returned, meaning it fails to retrieve 'accurate' regularity information.
        This tends to happen for short data segments, increasing tolerance
        levels might help avoid this.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, sampling_rate=100, frequency=[5, 6])

      # Range Entropy B (mSampEn)
      RangeEnB, info = nk.entropy_range(signal, approximate=False)
      RangeEnB

      # Range Entropy A (mApEn)
      RangeEnA, info = nk.entropy_range(signal, approximate=True)
      RangeEnA

      # Range Entropy A (corrected)
      RangeEnAc, info = nk.entropy_range(signal, approximate=True, corrected=True)
      RangeEnAc

    References
    ----------
    * Omidvarnia, A., Mesbah, M., Pedersen, M., & Jackson, G. (2018). Range entropy: A bridge
      between signal complexity and self-similarity. Entropy, 20(12), 962.


    """
    if approximate is False:  # mSampEn - RangeEn (B)
        out = entropy_sample(
            signal,
            delay=delay,
            dimension=dimension,
            tolerance=tolerance,
            distance="range",
            **kwargs,
        )
    else:  # mApEn - RangeEn (A)
        out = entropy_approximate(
            signal,
            delay=delay,
            dimension=dimension,
            tolerance=tolerance,
            distance="range",
            **kwargs,
        )
    return out
