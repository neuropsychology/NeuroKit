import numpy as np
import pandas as pd
import scipy.stats

from .fractal_petrosian import _complexity_binarize


def entropy_shannon(signal, base=2, method=None, show=False):
    """**Shannon entropy (SE or ShanEn)**

    Compute Shannon entropy (SE). Entropy is a measure of unpredictability of the
    state, or equivalently, of its average information content. Shannon entropy (SE) is one of the
    first and most basic measure of entropy and a foundational concept of information theory,
    introduced by Shannon (1948) to quantify the amount of information in a variable.

    Shannon attempted to extend Shannon entropy in what has become known as Differential Entropy
    (see ``entropy_differential()``).

    Because Shannon entropy was meant for symbolic sequences (discrete events such as ["A", "B",
    "B", "A"]), it does not do well with continuous signals. One option is to binarize (i.e., cut)
    the signal into a number of bins using for instance ``pd.cut(signal, bins=100, labels=False)``.
    This can be done automatically using the ``method`` argument, which is the same as in
    :func:`fractal_petrosian`. This means that methods *A*, *B*, *C*, *D*, and *r* are also
    available.

    This function can be called either via ``entropy_shannon()`` or ``complexity_se()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    base: float
        The logarithmic base to use, defaults to ``2``. Note that ``scipy.stats.entropy()``
        uses ``np.e`` as default (the natural logarithm).
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
    entropy_differential, entropy_cumulative_residual

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal with Laplace Noise
      signal = nk.signal_simulate(duration=2, frequency=5, noise=0.01)

      # Compute Shannon's Entropy
      @savefig p_entropy_shannon1.png scale=100%
      shanen, info = nk.entropy_shannon(signal, method=3, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      shanen

    Compare with ``scipy`` (using the same base).

    .. ipython:: python

      import scipy.stats

      # Make the binning ourselves
      binned = pd.cut(signal, bins=3, labels=False)

      scipy.stats.entropy(pd.Series(binned).value_counts())
      shanen, info = nk.entropy_shannon(binned, base=np.e)
      shanen

    References
    -----------
    * Shannon, C. E. (1948). A mathematical theory of communication. The Bell system technical
      journal, 27(3), 379-423.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Check if string ('ABBA'), and convert each character to list (['A', 'B', 'B', 'A'])
    if isinstance(signal, str):
        signal = list(signal)

    # Force to array
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    # Make discrete
    if np.isscalar(signal) is False:
        signal, _ = _complexity_binarize(signal, method=method, show=show)

    shanen = scipy.stats.entropy(pd.Series(signal).value_counts(), base=base)

    return shanen, {"Base": base, "Method": method}
