import numpy as np
import pandas as pd

from ..stats import rescale


def fractal_sevcik(signal):
    """**Sevcik Fractal Dimension (SFD)**

    The SFD algorithm was proposed to calculate the fractal dimension of waveforms by Sevcik
    (1998). This method can be used to quickly measure the complexity and randomness of a signal.

    The waveform is first mapped into a unit square (the time axis is rescaled to [0, 1] and the
    amplitude to [0, 1]), and its length *L* is computed. The fractal dimension is then
    approximated as :math:`D = 1 + \\frac{\\ln(L)}{\\ln(2 \\cdot N')}`, where *N'* is the number of
    segments (i.e., the number of samples minus 1).

    .. note:: Some papers (e.g., Wang et al. 2017) include a ``ln(2)`` term in the numerator. It
       comes from Sevcik's derivation, in which the curve is covered by :math:`N(\\epsilon) =
       L / (2 \\epsilon)` balls of radius :math:`\\epsilon = 1 / (2 N')`, giving
       :math:`D = 1 + \\frac{\\ln(L) - \\ln(2)}{\\ln(2 N')}`. Sevcik drops it because it vanishes as
       :math:`N' \\to \\infty` (his eq. 6), and that is the version that reproduces the values of his
       Table 2. We use eq. 6 (note that the QuickBASIC program listed in his appendix subtracts
       ``ln(2)``).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    Returns
    ---------
    sfd : float
        The sevcik fractal dimension.
    info : dict
        An empty dictionary returned for consistency with the other complexity functions.

    See Also
    --------
    fractal_petrosian

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=5)

      sfd, _ = nk.fractal_sevcik(signal)
      sfd

    References
    ----------
    * Sevcik, C. (2010). A procedure to estimate the fractal dimension of waveforms. arXiv preprint
      arXiv:1003.5266.
    * Kumar, D. K., Arjunan, S. P., & Aliahmad, B. (2017). Fractals: applications in biological
      Signalling and image processing. CRC Press.
    * Wang, H., Li, J., Guo, L., Dou, Z., Lin, Y., & Zhou, R. (2017). Fractal complexity-based
      feature extraction algorithm of communication signals. Fractals, 25(04), 1740008.
    * Goh, C., Hamadicharef, B., Henderson, G., & Ifeachor, E. (2005, June). Comparison of fractal
      dimension algorithms for the computation of EEG biomarkers for dementia. In 2nd International
      Conference on Computational Intelligence in Medicine and Healthcare (CIMED2005).

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    # 1. Normalize the signal (new range to [0, 1])
    y_ = rescale(signal, to=[0, 1])
    n = len(y_)

    # 2. Derive x* and y* (y* is actually the normalized signal)
    x_ = np.linspace(0, 1, n)

    # 3. Compute L (because we use np.diff, hence n-1 below)
    L = np.sum(np.sqrt(np.diff(y_) ** 2 + np.diff(x_) ** 2))

    # 4. Compute the fractal dimension (approximation, eq. 6 of Sevcik, 2010)
    sfd = 1 + np.log(L) / np.log(2 * (n - 1))

    # Sevcik's derivation (eq. 3) has an additional -ln(2) term in the numerator, which vanishes as
    # n -> Inf: sfd = 1 + (np.log(L) - np.log(2)) / np.log(2 * (n - 1)). Eq. 6 (without it) is the
    # one that reproduces the values in Sevcik's Table 2.

    return sfd, {}
