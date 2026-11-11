from warnings import warn

import numpy as np
import pandas as pd
import scipy.special

from ..misc import NeuroKitWarning
from .utils_complexity_embedding import complexity_embedding
from .utils_complexity_symbolize import complexity_symbolize


def entropy_symbolicdynamic(signal, dimension=3, symbolize="MEP", c=6, fluctuation=False, **kwargs):
    """**Symbolic Dynamic Entropy (SyDyEn) and its Multiscale variants (MSSyDyEn)**

    Symbolic Dynamic Entropy (SyDyEn) combines the merits of symbolic dynamic and information
    theory.

    The signal is first converted into a sequence of *c* symbols (by default with the maximum
    entropy partitioning, MEP, see :func:`complexity_symbolize`). The vectors of *m*
    (``dimension``) symbols, spaced by ``delay`` samples, are the *states* *q*, of probabilities
    :math:`P(q)`. The symbol :math:`\\sigma` that follows each vector (*m* × ``delay`` samples
    after its first one) is its *state transition*, and :math:`P(\\sigma | q)` is the proportion of
    the occurrences of the state *q* that are followed by the symbol :math:`\\sigma`. SyDyEn is the
    sum of the state entropy and of the state transition entropy, i.e., the entropy of the
    transitions conditional on the state (Li et al., 2017; Wei et al., 2023, eq. 6):

    .. math::

      SyDyEn = -\\sum_{q} P(q) \\ln P(q) - \\sum_{q} \\sum_{\\sigma} P(q) P(\\sigma | q) \\ln P(\\sigma | q)

    It is therefore the joint entropy of the states and of their transitions (i.e., of the
    sequences of *m + 1* symbols), and it is divided by its maximum, :math:`\\ln(c^{m+1})` (Wei et
    al., 2023, eq. 7), so that it lies between 0 and 1 (1 when all these sequences are equally
    likely, e.g., for white noise with the default MEP symbolization). The number of possible
    symbols *c* is 2 for the binary symbolizations (e.g., ``symbolize="A"``), whatever ``c``, and
    the number of bins if ``symbolize`` is an integer.

    Some implementations (e.g., EntropyHub) use a transition term without the weight
    :math:`P(\\sigma | q)`, :math:`-\\sum_{q} \\sum_{\\sigma} P(q) \\ln(P(q) P(\\sigma | q))`, which is not
    an entropy: it grows with the number of different transitions observed, and the normalization
    does not bound it. Wei et al. (2023) note that the definition of Li et al. (2017) should be the
    one above, and Shen et al. (2025), from the same group, also compute the joint entropy of the
    states and of their transitions.

    * **Fluctuation-based SyDyEn** (FSDE; Shen et al., 2025) uses, as the fluctuation-based
      :func:`Dispersion Entropy <entropy_dispersion>`, the differences between the neighbouring
      symbols of each vector instead of the symbols themselves: the states are the vectors of
      *m - 1* differences (from :math:`-(c - 1)` to :math:`c - 1`), and the transition is the
      difference with the following symbol. It is divided by :math:`\\ln((2c - 1)^m)`, the maximum of
      the joint entropy of the states and of their transitions. Shen et al. (2025) divide it by
      :math:`\\ln((2c - 1)^{m - 1})`, the maximum of the state entropy alone, which gives values
      *m / (m - 1)* times larger (that can exceed 1). They use ``dimension=5`` and ``c=20``, and its
      multiscale version (MFSDE) is ``entropy_multiscale(method="MSSyDyEn", fluctuation=True)``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter. It should
        be at least 2 with ``fluctuation=True``.
    symbolize : str
        Method to convert a continuous signal input into a symbolic (discrete) signal. Can be one
        of ``"MEP"`` (default), ``"NCDF"``, ``"linear"``, ``"uniform"``, ``"kmeans"``, ``"equal"``,
        or others. See :func:`complexity_symbolize` for details.
    c : int
        Number of symbols *c* (for the symbolizations that take it, such as ``"MEP"``). Li et al.
        (2017) used 8 (with ``dimension=3``; Wei et al., 2023).
    fluctuation : bool
        Fluctuation-based Symbolic Dynamic Entropy (FSDE).
    **kwargs : optional
        Other keyword arguments, such as the ``delay`` between the samples of the vectors (1 by
        default), or the ``random_state`` of :func:`complexity_symbolize` (e.g., for
        ``symbolize="kmeans"``).

    Returns
    -------
    SyDyEn : float
        Symbolic Dynamic Entropy (SyDyEn) of the signal. It is NaN (with a warning) if the signal
        contains missing values.
    info : dict
        A dictionary containing additional information regarding the parameters used.

    See Also
    --------
    entropy_shannon, entropy_multiscale, entropy_dispersion

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal
      signal = nk.signal_simulate(duration=2, sampling_rate=200, frequency=[5, 6], noise=0.5)

      # Compute Symbolic Dynamic Entropy
      sydyen, info = nk.entropy_symbolicdynamic(signal, c=3, symbolize="MEP")
      sydyen

      sydyen, info = nk.entropy_symbolicdynamic(signal, c=3, symbolize="kmeans", random_state=42)
      sydyen

      # Fluctuation-based Symbolic Dynamic Entropy (FSDE)
      fsde, info = nk.entropy_symbolicdynamic(signal, c=3, fluctuation=True)
      fsde

      # Compute Multiscale Symbolic Dynamic Entropy (MSSyDyEn)
      @savefig p_entropy_symbolicdynamic1.png scale=100%
      mssydyen, info = nk.entropy_multiscale(signal, method="MSSyDyEn", show=True)
      @suppress
      plt.close()

      # Compute Modified Multiscale Symbolic Dynamic Entropy (MMSyDyEn)
      @savefig p_entropy_symbolicdynamic2.png scale=100%
      mmsydyen, info = nk.entropy_multiscale(signal, method="MMSyDyEn", show=True)
      @suppress
      plt.close()

    References
    ----------
    * Matilla-García, M., Morales, I., Rodríguez, J. M., & Marín, M. R. (2021). Selection of
      embedding dimension and delay time in phase space reconstruction via symbolic dynamics.
      Entropy, 23(2), 221.
    * Li, Y., Yang, Y., Li, G., Xu, M., & Huang, W. (2017). A fault diagnosis scheme for planetary
      gearboxes using modified multi-scale symbolic dynamic entropy and mRMR feature selection.
      Mechanical Systems and Signal Processing, 91, 295-312.
    * Wei, Y., Wang, X., Xu, Y., & Fan, F. (2023). Intelligent fault diagnosis of rotating machinery
      using composite multivariate-based multi-scale symbolic dynamic entropy with multi-source
      monitoring data. Structural Health Monitoring, 22(1), 56-77.
    * Shen, A., Li, Y., Noman, K., Wang, D., Peng, Z., & Feng, K. (2025). Multiscale
      fluctuation-based symbolic dynamic entropy: a novel entropy method for fault diagnosis of
      rotating machinery. Structural Health Monitoring, 24(1), 402-420.
    * Rajagopalan, V., & Ray, A. (2006). Symbolic time series analysis via wavelet-based
      partitioning. Signal processing, 86(11), 3309-3320.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    if dimension < 2 and fluctuation is True:
        raise ValueError("The dimension of fluctuation-based Symbolic Dynamic Entropy should be at least 2.")

    # The delay can be passed as a keyword argument (e.g., by entropy_multiscale())
    delay = kwargs.pop("delay", 1)

    # Store parameters
    info = {"Dimension": dimension, "Delay": delay, "c": c, "Symbolization": symbolize, "Fluctuation": fluctuation}

    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if np.any(np.isnan(signal)):
        warn(
            "The signal contains missing values, which cannot be symbolized. Returning NaN.",
            category=NeuroKitWarning,
        )
        return np.nan, info
    if n <= dimension * delay:
        raise ValueError(
            "The signal is too short for Symbolic Dynamic Entropy, which needs at least one state"
            " transition (i.e., more than dimension * delay samples)."
        )

    # There are four main steps of SDE algorithm
    # 1. Convert the time series into the symbol time series (called symbolization). The symbols
    # are then replaced by their rank (0 to the number of different symbols - 1)
    symbolic = complexity_symbolize(signal, method=symbolize, c=c, random_state=kwargs.get("random_state"))

    # Fluctuation (Shen et al., 2025): the differences between the symbols spaced by the delay, from
    # -(c - 1) to c - 1, of which the states are the vectors of m - 1 (the m symbols of each vector)
    if fluctuation is True:
        symbolic = np.reshape(symbolic, -1)
        symbolic = symbolic[delay:] - symbolic[:-delay]
        dimension = dimension - 1

    _, symbolic = np.unique(symbolic, return_inverse=True)
    symbolic = np.reshape(symbolic, -1)
    n_symbols = np.max(symbolic) + 1

    # 2. Construct the embedding vectors based on the symbol time series and compute the potential
    # state patterns probability
    embedded = complexity_embedding(symbolic, dimension=dimension, delay=delay)
    _, states, counts = np.unique(embedded, axis=0, return_inverse=True, return_counts=True)
    states = np.reshape(states, -1)
    p_states = counts / len(embedded)

    # 3. Construct the state transitions and compute the probability of state transitions: the
    # symbol that follows each state, at j + dimension * delay (for all the j < n - dimension * delay)
    following = symbolic[dimension * delay :]
    transitions = np.bincount(
        states[: len(following)] * n_symbols + following,
        minlength=len(counts) * n_symbols,
    ).reshape(len(counts), n_symbols)
    total = np.sum(transitions, axis=1, keepdims=True)
    p_transitions = transitions / np.maximum(total, 1)

    # 4. SDE is the sum of the state entropy and of the state transition entropy, the entropy of the
    # symbol that follows a state conditional on that state (i.e., the joint entropy of both)
    state_entropy = np.sum(scipy.special.entr(p_states))
    transition_entropy = np.sum(p_states * np.sum(scipy.special.entr(p_transitions), axis=1))
    sydyen = state_entropy + transition_entropy

    # Normalize by the maximum, ln(c^(m+1)), with c the number of possible symbols: 2 for the binary
    # symbolizations (whatever c), and the number of bins when `symbolize` is an integer
    if isinstance(symbolize, (int, np.integer)):
        n_possible = int(symbolize)
    elif isinstance(symbolize, str) and symbolize.lower() in ["a", "mean", "median", "b", "c", "sign", "d", "r"]:
        n_possible = 2
    else:
        n_possible = c
    # The differences between c symbols can take 2c - 1 values (and the states have m - 1 of them)
    if fluctuation is True:
        n_possible = 2 * n_possible - 1
    sydyen = sydyen / ((dimension + 1) * np.log(max(n_possible, n_symbols)))

    return sydyen, info
