from warnings import warn

import numpy as np

from ..misc import NeuroKitWarning
from .utils_complexity_embedding import complexity_embedding
from .utils_complexity_symbolize import complexity_symbolize


def information_gain(signal, delay=1, dimension=4, symbolize="mean"):
    """**Mean Information Gain (MIG)** and **Fluctuation Complexity (FC)**

    Mean Information Gain (MIG) is a measure of diversity, as it exhibits maximum values for random
    signals. (Bates & Shepard, 1993; Wackerbauer et al., 1994).

    Unlike MIG, fluctuation complexity (FC) does not consider a random signal to be complex. The
    fluctuation complexity is the mean square deviation of the net information gain (i.e. the
    differences between information gain and loss). The more this balance of information gain and
    loss is fluctuating, the more complex the signal is considered.

    It is to note that the original formulations discuss the length of the "words" (the number of
    samples) the signal is partitioned into, this length parameter corresponds to the embedding
    dimension (i.e., the amount of past states to consider, by default 4). We additionally modified
    the original algorithm to the possibility of modulating the delay between each past state.

    The symbolized signal is cut into overlapping words *i* of ``dimension`` symbols (``delay``
    samples apart), each followed by the word *j* that starts ``delay`` samples later (and therefore
    adds one new symbol). With :math:`p_i` the probability of word *i*, :math:`p_{ij}` the
    probability of the transition from *i* to *j*, and :math:`p_{j|i} = p_{ij} / p_i` the
    probability of *j* given *i*:

    .. math::

      MIG = -\\sum_{i,j} p_{ij} \\log_2 p_{j|i}

      FC = \\sum_{i,j} p_{ij} \\left(\\log_2 \\frac{p_i}{p_j}\\right)^2

    MIG (in bits) is the conditional entropy of the next word given the current one, i.e., the
    entropy of the words of ``dimension + 1`` symbols minus that of the words of ``dimension``
    symbols. It is 1 bit for a random binary signal (the default symbolization), whatever the
    ``dimension`` and ``delay``.

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
    symbolize : str
        Method to convert a continuous signal input into a symbolic (discrete) signal. By default,
        assigns 0 and 1 to values below and above the mean. Can be ``None`` to skip the process (in
        case the input is already discrete). See :func:`complexity_symbolize` for details.

    Returns
    -------
    mig : float
        The Mean Information Gain value (MIG).
    info : dict
        A dictionary containing additional information, such as the Fluctuation Complexity
        (``"FC"``).

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=[5, 12, 85])

      mig, info = nk.information_gain(signal)

      # Mean Information Gain (MIG)
      mig

      # Fluctuation Complexity
      info['FC']


    References
    ----------
    * Bates, J. E., & Shepard, H. K. (1993). Measuring complexity using information fluctuation.
      Physics Letters A, 172(6), 416-425.
    * Wackerbauer, R., Witt, A., Atmanspacher, H., Kurths, J., & Scheingraber, H. (1994). A
      comparative classification of complexity measures. Chaos, Solitons & Fractals, 4(1), 133-173.

    """
    info = {"Dimension": dimension, "Delay": delay}

    values = np.asarray(signal)
    if np.issubdtype(values.dtype, np.number) and np.any(np.isnan(values)):
        warn("The signal contains missing values, which cannot be symbolized. Returning NaN.", category=NeuroKitWarning)
        return np.nan, {**info, "FC": np.nan}

    # Discretize the signal (by default, into zeros and ones)
    symbolic = complexity_symbolize(signal, method=symbolize)

    # Replace the symbols by integers from 0 to k - 1, so that any symbols can be used (e.g., more
    # than 10 different values, or non-integer values)
    alphabet, codes = np.unique(symbolic, return_inverse=True)
    k = max(len(alphabet), 2)

    # Get overlapping windows of a given width
    embedded = complexity_embedding(codes.reshape(-1), dimension=dimension, delay=delay).astype(np.int64)

    # Label each word with an integer: its number in base k (or, when there are too many possible
    # words for 64-bit integers, by comparing the rows as a whole, which is slower)
    if dimension * np.log2(k) < 62:
        words = embedded @ (k ** np.arange(dimension, dtype=np.int64))
        _, states, counts = np.unique(words, return_inverse=True, return_counts=True)
    else:
        _, states, counts = np.unique(embedded, axis=0, return_inverse=True, return_counts=True)
    states = states.reshape(-1)
    n_states = len(counts)
    p_states = counts / len(states)

    # Transitions from each word to the next one, which starts ``delay`` samples later
    first, second = states[:-delay], states[delay:]
    if len(first) == 0:
        warn("The signal is too short to have a transition between words. Returning NaN.", category=NeuroKitWarning)
        return np.nan, {**info, "FC": np.nan}
    transitions, n_ij = np.unique(first * n_states + second, return_counts=True)
    i, j = np.divmod(transitions, n_states)
    p_ij = n_ij / len(first)

    # Information gain: log2(1 / p(j|i)), with p(j|i) = n(i, j) / n(i)
    n_i = np.bincount(first, minlength=n_states)
    mig = np.sum(p_ij * np.log2(n_i[i] / n_ij))
    # Net information gain
    fc = np.sum(p_ij * np.log2(p_states[i] / p_states[j]) ** 2)

    return mig, {**info, "FC": fc}
