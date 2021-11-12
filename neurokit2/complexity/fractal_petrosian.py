import numpy as np
import pandas as pd

from ..signal.signal_binarize import _signal_binarize_threshold


def fractal_petrosian(signal, method="C"):
    """Petrosian fractal dimension (PFD)

    Petrosian proposed a fast method to estimate the fractal dimension of a finite sequence, which
    converts the data to binary sequence before estimating the fractal dimension from time series.
    Several variations of the algorithm exist (e.g., 'A', 'B', 'C' or 'D'), primarily differing in
    the way the binary sequence is created.

    See Also
    --------
    mutual_information, entropy_svd

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        Can be 'A', 'B', 'C' or 'D'. Method 'A' binarizes the signal by higher vs. lower values as
        compated to the signal's mean. Method 'B' uses values that are within the mean +/- 1 SD band
        vs. values that are outside this band. Method 'C' computes the difference between consecutive
        samples and binarizes depending on their sign. Method 'D' forms separates consecutive samples
        that exceed 1 signal's SD from the others smaller changes.

    Returns
    -------
    pfd : float
        The petrosian fractal dimension (PFD).
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute PFD.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>>
    >>> pfd, info = nk.fractal_petrosian(signal, method = "A")
    >>> pfd, info = nk.fractal_petrosian(signal, method = "B")
    >>> pfd, info = nk.fractal_petrosian(signal, method = "C")
    >>> pfd, info = nk.fractal_petrosian(signal, method = "D")

    References
    ----------
    - Kumar, D. K., Arjunan, S. P., & Aliahmad, B. (2017). Fractals: applications in biological
    Signalling and image processing. CRC Press.
    - Goh, C., Hamadicharef, B., Henderson, G., & Ifeachor, E. (2005, June). Comparison of fractal
    dimension algorithms for the computation of EEG biomarkers for dementia. In 2nd International
    Conference on Computational Intelligence in Medicine and Healthcare (CIMED2005).

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Binarize the sequence
    if method == "A":
        n_inversions = _signal_binarize_threshold(signal, threshold="mean").sum()
    elif method == "B":
        range = (
            np.nanmean(signal) - np.nanstd(signal, ddof=1),
            np.nanmean(signal) + np.nanstd(signal, ddof=1),
        )
        n_inversions = np.logical_or(signal < range[0], signal > range[1]).sum()
    elif method == "C":
        # Method 1 (antropy and https://stackoverflow.com/a/29674950/10581531)
        n_inversions = np.diff(np.signbit(signal)).sum()

        # # Method 2 (https://github.com/gilestrolab/pyrem/blob/master/src/pyrem/univariate.py#L169)
        # diff = np.diff(signal)
        # prod = diff[1:-1] * diff[0:-2]  # x[i] * x[i-1] for i in t0 -> tmax
        # n_inversions = np.sum(prod < 0)
    elif method == "D":
        sd = np.nanstd(signal, ddof=1)
        diff = np.abs(np.diff(signal))
        n_inversions = (diff > sd).sum()
    else:
        raise ValueError(
            "method must be one of 'A', 'B', 'C' or 'D'. See the documentation for more information."
        )

    n = len(signal)
    pfd = np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_inversions)))
    return pfd, {"Method": method}
