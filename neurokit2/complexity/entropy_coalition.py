import numpy as np
import pandas as pd
import scipy.signal

from ..signal.signal_binarize import _signal_binarize_threshold
from ..signal.signal_detrend import signal_detrend
from .entropy_shannon import entropy_shannon


def entropy_coalition(signal, method="amplitude"):
    """**Amplitude Coalition Entropy (ACE) and Synchrony Coalition Entropy (SCE)**

    Amplitude Coalition Entropy (ACE) reflects the entropy over time of the constitution of the set
    of most active channels (Shanahan, 2010), and is similar to Lempel-Ziv complexity, in the sense
    that it quantifies variability in space and time of the activity. ACE is normalized by dividing
    the raw by the value obtained for the same binary input but randomly shuffled. The
    implementation used here is that of Schartner et al.'s (2015), which modified Shanahan's (2010)
    original version of coalition entropy so that it is applicable to real EEG data.

    Synchrony Coalition Entropy (SCE) reflects the entropy over time of the constitution of
    the set of synchronous channels, introduced and implemented by Schartner et al. (2015).
    SCE quantifies variability in the relationships between pairs of channel, i.e., the uncertainty
    over time of the constitution of the set of channels in synchrony (rather than active).
    The overall SCE is the mean value of SCE across channels.

    Parameters
    ----------
    signal : DataFrame
        The DataFrame containing all the respective signals (n_samples x n_channels).
    method : str
        Can be ``"amplitude"`` for ACE or ``"synchrony"`` for SCE.

    Returns
    ----------
    ce : float
         The coalition entropy.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute coalition entropy.

    References
    ----------
    * Shanahan, M. (2010). Metastable chimera states in community-structured oscillator networks.
      Chaos: An Interdisciplinary Journal of Nonlinear Science, 20(1), 013108.
    * Schartner, M., Seth, A., Noirhomme, Q., Boly, M., Bruno, M. A., Laureys, S., &
      Barrett, A. (2015). Complexity of multi-dimensional spontaneous EEG decreases
      during propofol induced general anaesthesia. PloS one, 10(8), e0133532.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Get data
      raw = nk.mne_data("raw")
      signal = nk.mne_to_df(raw)[["EEG 001", "EEG 002", "EEG 003"]]

      # ACE
      ace, info = nk.entropy_coalition(signal, method="amplitude")
      ace

      # SCE
      sce, info = nk.entropy_coalition(signal, method="synchrony")
      sce
    """
    # Sanity checks
    if isinstance(signal, pd.DataFrame):
        # return signal in (len(channels), len(samples)) format
        signal = signal.values.transpose()
    elif (isinstance(signal, np.ndarray) and len(signal.shape) == 1) or isinstance(
        signal, (list, pd.Series)
    ):
        raise ValueError(
            "entropy_coalition(): The input must be a dataframe containing multiple signals.",
        )

    # Detrend and normalize
    signal = np.array([signal_detrend(i - np.mean(i)) for i in signal])

    # Method
    method = method.lower()
    if method in ["ACE", "amplitude"]:
        info = {"Method": "ACE"}
        entropy = _entropy_coalition_amplitude(signal)
    elif method in ["SCE", "synchrony"]:
        info = {"Method": "SCE"}
        entropy, info["Values"] = _entropy_coalition_synchrony(signal)

    return entropy, info


# =============================================================================
# Methods
# =============================================================================


def _entropy_coalition_synchrony(signal):

    n_channels, n_samples = np.shape(signal)

    # Get binary matrices of synchrony for each series
    transformed = np.angle(scipy.signal.hilbert(signal))
    matrix = np.zeros(
        (n_channels, n_channels - 1, n_samples)
    )  # store array of synchrony series for each channel

    for i in range(n_channels):
        index = 0
        for j in range(n_channels):
            if i != j:
                matrix[i, index] = _entropy_coalition_synchrony_phase(
                    transformed[i], transformed[j]
                )
                index += 1

    # Create random binary matrix for normalization
    y = np.random.rand(n_channels - 1, n_samples)
    random_binarized = np.array([_signal_binarize_threshold(i, threshold=0.5) for i in y])
    norm = entropy_shannon(_entropy_coalition_map(random_binarized))[0]

    # Compute shannon entropy
    entropy = np.zeros(n_channels)
    for i in range(n_channels):
        c = _entropy_coalition_map(matrix[i])
        entropy[i] = entropy_shannon(c)[0]

    return np.mean(entropy) / norm, entropy / norm


def _entropy_coalition_amplitude(signal):

    # Hilbert transform to determine the amplitude envelope
    env = np.array([np.abs(scipy.signal.hilbert(i)) for i in signal])

    # Binarize (similar to LZC), mean of absolute of signal as threshold
    binarized = np.array([_signal_binarize_threshold(i, threshold="mean") for i in env])

    # Compute Shannon Entropy
    e1 = entropy_shannon(_entropy_coalition_map(binarized))[0]

    # Shuffle
    np.random.seed(30)  # set random seed to get reproducible results
    for seq in binarized:
        np.random.shuffle(seq)

    # Shuffled result as normalization
    e2 = entropy_shannon(_entropy_coalition_map(binarized))[0]

    return e1 / e2


# =============================================================================
# Utilities
# =============================================================================


def _entropy_coalition_synchrony_phase(phase1, phase2):
    """Compute synchrony of two series of phases"""
    diff = np.abs(phase1 - phase2)
    d2 = np.zeros(len(diff))
    for i in range(len(d2)):
        if diff[i] > np.pi:
            diff[i] = 2 * np.pi - diff[i]
        if diff[i] < 0.8:
            d2[i] = 1
    return d2


def _entropy_coalition_map(binary_sequence):
    """Map each binary column of binary matrix psi onto an integer"""
    n_channels, n_samples = binary_sequence.shape[0], binary_sequence.shape[1]

    mapped = np.zeros(n_samples)
    for t in range(n_samples):
        for j in range(n_channels):
            mapped[t] += binary_sequence[j, t] * (2 ** j)

    return mapped
