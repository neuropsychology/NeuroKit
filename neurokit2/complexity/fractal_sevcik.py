import numpy as np
import pandas as pd

from ..stats import rescale


def fractal_sevcik(signal):
    """Sevcik fractal dimension (SFD)

    The Sevcik algorithm was proposed to calculate the fractal dimension of waveforms by Sevcik (1998). This method could be
    used to quickly measure the complexity and randomness of a signal.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    Returns
    ---------
    sfd : float
        The sevcik fractal dimension.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute SFD.

    See Also
    --------
    fractal_petrosian

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>>
    >>> sfd, _ = nk.fractal_sevcik(signal)

    References
    ----------
    - Sevcik, C. (2010). A procedure to estimate the fractal dimension of waveforms. arXiv preprint arXiv:1003.5266.
    - Kumar, D. K., Arjunan, S. P., & Aliahmad, B. (2017). Fractals: applications in biological
    Signalling and image processing. CRC Press.
    - Wang, H., Li, J., Guo, L., Dou, Z., Lin, Y., & Zhou, R. (2017). Fractal complexity-based
    feature extraction algorithm of communication signals. Fractals, 25(04), 1740008.
    - Goh, C., Hamadicharef, B., Henderson, G., & Ifeachor, E. (2005, June). Comparison of fractal
    dimension algorithms for the computation of EEG biomarkers for dementia. In 2nd International
    Conference on Computational Intelligence in Medicine and Healthcare (CIMED2005).

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # 1. Normalize the signal (new range to [0, 1])
    y_ = rescale(signal, to=[0, 1])
    n = len(y_)

    # 2. Derive x* and y* (y* is actually the normalized signal)
    x_ = np.arange(1, n + 1) / n

    # 3. Compute L
    L = np.sum(np.sqrt(np.diff(y_) ** 2 + np.diff(x_) ** 2))

    # 4. Compute the fractal dimension (approximation)
    sfd = 1 + np.log(L) / np.log(2 * (n - 1))
    # Some papers (e.g., Wang et al. 2017) suggest adding np.log(2) to the numerator:
    # sfd = 1 + (np.log(L) + np.log(2)) / np.log(2 * (n - 1))
    # But it's unclear why. Sticking to the original formula for now.

    return sfd, {}
