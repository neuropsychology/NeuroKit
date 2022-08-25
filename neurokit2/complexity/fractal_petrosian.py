import numpy as np
import pandas as pd

from .utils_complexity_symbolize import complexity_symbolize


def fractal_petrosian(signal, symbolize="C", show=False):
    """**Petrosian fractal dimension (PFD)**

    Petrosian (1995) proposed a fast method to estimate the fractal dimension by converting the
    signal into a binary sequence from which the fractal dimension is estimated. Several variations
    of the algorithm exist (e.g., ``"A"``, ``"B"``, ``"C"`` or ``"D"``), primarily differing in the
    way the discrete (symbolic) sequence is created (see func:`complexity_symbolize` for details).
    The most common method (``"C"``, by default) binarizes the signal by the sign of consecutive
    differences.

    .. math::

      \\frac{log(N)}{log(N) + log(\\frac{N}{N+0.4N_{\\delta}})}

    Most of these methods assume that the signal is periodic (without a linear trend). Linear
    detrending might be useful to eliminate linear trends (see :func:`.signal_detrend`).

    See Also
    --------
    information_mutual, entropy_svd

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    symbolize : str
        Method to convert a continuous signal input into a symbolic (discrete) signal. By default,
        assigns 0 and 1 to values below and above the mean. Can be ``None`` to skip the process (in
        case the input is already discrete). See :func:`complexity_symbolize` for details.
    show : bool
        If ``True``, will show the discrete the signal.

    Returns
    -------
    pfd : float
        The petrosian fractal dimension (PFD).
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute PFD.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=[5, 12])

      @savefig p_fractal_petrosian1.png scale=100%
      pfd, info = nk.fractal_petrosian(signal, symbolize = "C", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      pfd
      info


    References
    ----------
    * Esteller, R., Vachtsevanos, G., Echauz, J., & Litt, B. (2001). A comparison of waveform
      fractal dimension algorithms. IEEE Transactions on Circuits and Systems I: Fundamental Theory
      and Applications, 48(2), 177-183.
    * Petrosian, A. (1995, June). Kolmogorov complexity of finite sequences and recognition of
      different preictal EEG patterns. In Proceedings eighth IEEE symposium on computer-based
      medical systems (pp. 212-217). IEEE.
    * Kumar, D. K., Arjunan, S. P., & Aliahmad, B. (2017). Fractals: applications in biological
      Signalling and image processing. CRC Press.
    * Goh, C., Hamadicharef, B., Henderson, G., & Ifeachor, E. (2005, June). Comparison of fractal
      dimension algorithms for the computation of EEG biomarkers for dementia. In 2nd International
      Conference on Computational Intelligence in Medicine and Healthcare (CIMED2005).

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Binarize the sequence
    symbolic = complexity_symbolize(signal, method=symbolize, show=show)

    # if isinstance(method, str) and method.lower() in ["d", "r"]:
    #     # These methods are already based on the consecutive differences
    #     n_inversions = symbolic.sum()
    # else:
    #     # Note: np.diff(symbolic).sum() wouldn't work in case there's a seq like [0, -1, 1]
    #     n_inversions = (symbolic[1:] != symbolic[:-1]).sum()
    n_inversions = (symbolic[1:] != symbolic[:-1]).sum()

    n = len(symbolic)
    pfd = np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_inversions)))
    return pfd, {"Symbolization": symbolize}
