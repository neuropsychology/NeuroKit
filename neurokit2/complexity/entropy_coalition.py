import numpy as np

from .entropy_shannon import entropy_shannon
from ..signal.signal_binarize import _signal_binarize_threshold
from ..signal.signal_detrend import signal_detrend


def entropy_coalition(signal):
    """Amplitude Coalition Entropy (ACE) reflects the entropy over time of the constitution of the set of
    most active channels, and is similar to Lempel-Ziv complexity, in the sense that it quantifies
    variability in space and time of the activity.
   
    Synchrony Coalition Entropy (SCE) reflects the entropy over time of the constitution of
    the set of synchronous channels. SCE quantifies variability in the relationships between pairs of channels.

    References
    ----------
    - Shanahan, M. (2010). Metastable chimera states in community-structured oscillator networks.
    Chaos: An Interdisciplinary Journal of Nonlinear Science, 20(1), 013108.

    TODO: Check implementation
   """

    # Detrend and normalize
    signal = np.array(signal).T  # must be multidimensional of (signal, n_samples) 
    detrended = np.array([signal_detrend(i - np.mean(i)) for i in signal])

    # Binarize (similar to LZC)
    binary_sequence = np.array([_signal_binarize_threshold(np.abs(i), threshold="mean") for i in detrended])

    # Map each binary column of binary matrix psi onto an integer
    mapped = np.zeros(binary_sequence.shape[1])
    for t in range(binary_sequence.shape[1]):
         for j in range(binary_sequence.shape[0]):
             mapped[t] += binary_sequence[j, t] * (2**j)

    # Compute Shannon Entropy
    e1 = entropy_shannon(_entropy_caolition_map(binary_sequence))[0]

    # Shuffle
    for seq in binary_sequence:
        np.random.shuffle(seq)
    e2 = entropy_shannon(_entropy_caolition_map(binary_sequence))[0]

    return e1 / e2


def _entropy_caolition_map(binary_sequence):
    # Map each binary column of binary matrix psi onto an integer
    mapped = np.zeros(binary_sequence.shape[1])
    for t in range(binary_sequence.shape[1]):
         for j in range(binary_sequence.shape[0]):
             mapped[t] += binary_sequence[j, t] * (2**j)
    return mapped
