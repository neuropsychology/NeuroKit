import numpy as np
import pandas as pd
import scipy.stats

from .utils_complexity_symbolize import complexity_symbolize


def entropy_shannon(signal=None, base=2, symbolize=None, show=False, freq=None, **kwargs):
    """**Shannon entropy (SE or ShanEn)**

    Compute Shannon entropy (SE). Entropy is a measure of unpredictability of the
    state, or equivalently, of its average information content. Shannon entropy (SE) is one of the
    first and most basic measures of entropy and a foundational concept of information theory,
    introduced by Shannon (1948) to quantify the amount of information in a variable.

    .. math::

      ShanEn = -\\sum_{x \\in \\mathcal{X}} p(x) \\log_b p(x)

    where *p(x)* is the proportion of the samples that are equal to the state *x* (states that do
    not occur do not contribute), and *b* is the logarithmic ``base`` (2 by default, giving bits).

    Shannon attempted to extend Shannon entropy in what has become known as Differential Entropy
    (see :func:`entropy_differential`).

    Because Shannon entropy was meant for symbolic sequences (discrete events such as ["A", "B",
    "B", "A"]), it does not do well with continuous signals: every distinct value is then a state.
    One option is to binarize (i.e., cut) the signal into a number of bins using for instance
    ``pd.cut(signal, bins=100, labels=False)``. This can be done automatically using the
    ``symbolize`` argument, which will be transferred to :func:`complexity_symbolize`.

    This function can be called either via ``entropy_shannon()`` or ``complexity_se()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series, str]
        The signal (i.e., a time series) in the form of a vector of values. A string (e.g.,
        ``"ABBA"``) is treated as a sequence of characters.
    base : float
        The logarithmic base to use, defaults to ``2``, giving a unit in *bits*. Note that ``scipy.
        stats.entropy()`` uses Euler's number (``np.e``) as default (the natural logarithm), giving
        a measure of information expressed in *nats*.
    symbolize : str
        Method to convert a continuous signal input into a symbolic (discrete) signal. ``None`` by
        default, which skips the process (and assumes the input is already discrete). See
        :func:`complexity_symbolize` for details.
    show : bool
        If ``True``, will show the discrete signal.
    freq : np.array
        Instead of a signal, a vector of probabilities (or counts, which are normalized) of each
        state can be provided (used for instance in :func:`entropy_permutation`).
    **kwargs
        Other arguments passed to :func:`complexity_symbolize`, such as ``c`` (the number of
        symbols for the methods that need it) or ``random_state``.


    Returns
    --------
    shanen : float
        The Shannon entropy of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute Shannon entropy.

    See Also
    --------
    entropy_differential, entropy_cumulativeresidual, entropy_tsallis, entropy_renyi,
    entropy_maximum

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = [1, 1, 5, 5, 2, 8, 1]
      _, freq = np.unique(signal, return_counts=True)
      nk.entropy_shannon(freq=freq)

    .. ipython:: python

      # Simulate a Signal with Laplace Noise
      signal = nk.signal_simulate(duration=2, frequency=5, noise=0.01)

      # Compute Shannon's Entropy
      @savefig p_entropy_shannon1.png scale=100%
      shanen, info = nk.entropy_shannon(signal, symbolize=3, show=True)
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
    if freq is None:
        _, freq = _entropy_freq(signal, symbolize=symbolize, show=show, **kwargs)

    return scipy.stats.entropy(freq, base=base), {"Symbolization": symbolize, "Base": base}


# =============================================================================
# Compute frequencies (common to Shannon, Rényi, Tsallis and CREn)
# =============================================================================
def _entropy_freq(signal, symbolize=None, show=False, **kwargs):
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    # Check if string ('ABBA'), and convert each character to list (['A', 'B', 'B', 'A'])
    if isinstance(signal, str):
        signal = list(signal)

    # Force to array
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    # A numpy integer (e.g., np.int64(3)) is a number of bins, like a Python integer
    if isinstance(symbolize, np.integer):
        symbolize = int(symbolize)

    # Make discrete (passing on the arguments of complexity_symbolize, such as the number of symbols)
    options = {key: kwargs[key] for key in ["c", "random_state"] if key in kwargs}
    signal = complexity_symbolize(signal, method=symbolize, show=show, **options)

    return np.unique(signal, return_counts=True)
