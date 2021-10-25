import numpy as np
import scipy.signal

from .entropy_shannon import entropy_shannon
from ..signal.signal_binarize import _signal_binarize_threshold
from ..signal.signal_detrend import signal_detrend
from .utils import _sanitize_multichannel


def entropy_coalition(signal, method="amplitude"):
    """
    Amplitude Coalition Entropy (ACE) reflects the entropy over time of the constitution of the set of
    most active channels (Shanahan, 2010), and is similar to Lempel-Ziv complexity, in the sense that it quantifies
    variability in space and time of the activity. ACE is normalized by dividing the raw  by the value
    obtained for the same binary input but randomly shuffled. The implementation used here is that of Schartner
    et al.'s (2015), which modified Shanahan's (2010) original version of coalition entropy so that it is
    applicable to real EEG data.
 
    Synchrony Coalition Entropy (SCE) reflects the entropy over time of the constitution of
    the set of synchronous channels, introduced and implemented by Schartner et al. (2015).
    SCE quantifies variability in the relationships between pairs of channel, i.e., the uncertainty
    over time of the constitution of the set of channels in synchrony (rather than active).
    The overall SCE is the mean value of SCE across channels.

    References
    ----------
    - Shanahan, M. (2010). Metastable chimera states in community-structured oscillator networks.
    Chaos: An Interdisciplinary Journal of Nonlinear Science, 20(1), 013108.

    - Schartner, M., Seth, A., Noirhomme, Q., Boly, M., Bruno, M. A., Laureys, S., & Barrett, A. (2015).
    Complexity of multi-dimensional spontaneous EEG decreases during propofol induced general anaesthesia.
    PloS one, 10(8), e0133532.

    TODO: Check implementation
    
    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Get data
    >>> raw = nk.mne_data("raw")
    >>> signal = nk.mne_to_df(raw)[["EEG 001", "EEG 002"]].iloc[0:5000]
    >>>
    >>> # ACE
    >>> ace, info = nk.entropy_coalition(signal, method="amplitude")
    >>> ace #doctest: +SKIP
   """
    # Sanity checks
    signal = _sanitize_multichannel(signal)

    # Method
    method = method.lower()
    if method in ["ACE", "amplitude"]:
        info = {"Method": "ACE"}
        entropy = _entropy_coalition_amplitude(signal)
    elif method in ["SCE", "sychrony"]:
        info = {"Method": "SCE"}
        entropy = _entropy_coalition_synchrony(signal)

    return entropy, info

# =============================================================================
# Methods
# =============================================================================

def _entropy_coalition_synchrony(signal):

    return "TEST"

def _entropy_coalition_amplitude(signal):

    # Detrend and normalize
    detrended = np.array([signal_detrend(i - np.mean(i)) for i in signal])

    # Hilbert transform to determine the amplitude envelope
    env = np.array([np.abs(scipy.signal.hilbert(i)) for i in detrended])

    # Binarize (similar to LZC), mean of absolute of signal as threshold
    binarized = np.array([_signal_binarize_threshold(i, threshold="mean") for i in env])

    # Compute Shannon Entropy
    e1 = entropy_shannon(_entropy_coalition_map(binarized))[0]

    # Shuffle
    np.random.seed(30)  # set random seed to get reproducible results
    new_binarized = [np.random.shuffle(seq) for seq in binarized]

    # Shuffled result as normalization
    e2 = entropy_shannon(_entropy_coalition_map(binarized))[0]

    return e1 / e2


# =============================================================================
# Utilities
# =============================================================================

def _entropy_coalition_map(binary_sequence):
    # Map each binary column of binary matrix psi onto an integer
    mapped = np.zeros(binary_sequence.shape[1])
    for t in range(binary_sequence.shape[1]):
         for j in range(binary_sequence.shape[0]):
             mapped[t] += binary_sequence[j, t] * (2**j)

    return mapped
