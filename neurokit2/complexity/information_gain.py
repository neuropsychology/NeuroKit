import numpy as np

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
        A dictionary containing additional information .

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
    # Discretize the signal into zeros and ones
    binary = complexity_symbolize(signal, method=symbolize)

    # Get overlapping windows of a given width
    embedded = complexity_embedding(binary, dimension=dimension, delay=delay).astype(int)

    # Convert into strings
    states = ["".join(list(state)) for state in embedded.astype(str)]
    transitions = [tuple(states[i : i + 2]) for i in range(len(states) - 1)]

    # Get unique and format
    states_unique, states_prob = np.unique(states, axis=0, return_counts=True)
    states_prob = states_prob / np.sum(states_prob)
    s_prob = {k: states_prob[i] for i, k in enumerate(states_unique)}

    transitions_unique, transitions_prob = np.unique(transitions, axis=0, return_counts=True)
    transitions_prob = transitions_prob / np.sum(transitions_prob)
    t_prob = {tuple(k): transitions_prob[i] for i, k in enumerate(transitions_unique)}

    mig = 0
    fc = 0
    for i in states_unique:
        for j in states_unique:

            if (i, j) not in t_prob.keys():
                continue
            i_j_prob = t_prob.get((i, j), 0)
            if i_j_prob == 0:
                continue

            # Information gain
            mig += i_j_prob * np.log2(1 / (i_j_prob / s_prob.get(i)))

            # Net information gain
            fc += i_j_prob * (np.log2(s_prob[i] / s_prob[j]) ** 2)
    return mig, {"Dimension": dimension, "Delay": delay, "FC": fc}
