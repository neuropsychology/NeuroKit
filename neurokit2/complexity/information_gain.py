import numpy as np

from .utils_complexity_embedding import complexity_embedding
from .utils_complexity_symbolize import complexity_symbolize


def information_gain(signal, width=4):
    """**Mean Information Gain (MIG)**

    Mean Information Gain (MIG) is ...

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    width : int
        The length of the verbs created (default to 4 samples) that will correspond to discrete
        states.

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

      mig, info = nk.information_gain(signal, width=4)
      mig

    References
    ----------
    * Bates, J. E., & Shepard, H. K. (1993). Measuring complexity using information fluctuation.
      Physics Letters A, 172(6), 416-425.
    * Wackerbauer, R., Witt, A., Atmanspacher, H., Kurths, J., & Scheingraber, H. (1994). A
      comparative classification of complexity measures. Chaos, Solitons & Fractals, 4(1), 133-173.

    """

    binary = complexity_symbolize(signal, method="median")

    # Get overlapping windows of a given width
    embedded = complexity_embedding(binary, dimension=width, delay=1).astype(int)

    # Convert into strings
    states = ["".join(list(l)) for l in embedded.astype(str)]
    transitions = [tuple(states[i : i + 2]) for i in range(len(states) - 1)]

    # Get unique and format
    states_unique, states_prob = np.unique(states, axis=0, return_counts=True)
    states_prob = states_prob / np.sum(states_prob)
    s_prob = {k: states_prob[i] for i, k in enumerate(states_unique)}

    transitions_unique, transitions_prob = np.unique(transitions, axis=0, return_counts=True)
    transitions_prob = transitions_prob / np.sum(transitions_prob)
    t_prob = {tuple(k): transitions_prob[i] for i, k in enumerate(transitions_unique)}

    mig = 0
    for i in states_unique:
        for j in states_unique:
            mig += t_prob.get((i, j), 0.0) * _information_gain(i, j, t_prob, s_prob)
    return mig, {"width": width}


# -------------------------------------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------------------------------------
def _information_gain(i, j, t_prob, s_prob):
    if (i, j) not in t_prob.keys():
        return 0
    i_j_prob = t_prob.get((i, j), 0.0) / s_prob.get(i)
    if i_j_prob == 0:
        return 0
    return np.log2(1 / i_j_prob)
