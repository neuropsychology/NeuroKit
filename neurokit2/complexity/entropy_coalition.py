from warnings import warn

import numpy as np
import pandas as pd
import scipy.signal

from ..misc import NeuroKitWarning, check_random_state
from ..signal.signal_binarize import _signal_binarize_threshold
from ..signal.signal_detrend import signal_detrend
from .entropy_shannon import entropy_shannon


def entropy_coalition(signal, method="amplitude", random_state=30):
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

    Each channel is first linearly detrended and its mean removed. For ACE, the amplitude envelope
    of each channel (the absolute value of its analytic signal, obtained by the Hilbert transform)
    is binarized by its mean, and the coalition at each time point (the set of channels above
    their mean) is a state, whose :func:`Shannon entropy <entropy_shannon>` over time is divided
    by that of the same binary sequences shuffled in time (independently for each channel). For
    SCE, two channels are synchronous at a time point if the absolute difference between their
    instantaneous phases (from the Hilbert transform) is less than 0.8 radians; for each channel,
    the entropy of the coalitions of the other channels that are synchronous with it is divided by
    that of random binary sequences (each entry being 1 with a probability of 0.5) of the same
    size.

    Parameters
    ----------
    signal : DataFrame
        The DataFrame containing all the respective signals (n_samples x n_channels), or an array
        of shape (n_channels, n_samples), with at least 2 channels.
    method : str
        Can be ``"amplitude"`` (or ``"ACE"``) for ACE or ``"synchrony"`` (or ``"SCE"``) for SCE.
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator used for the normalization (the shuffling of the
        binary sequences for ACE, or the random binary sequences for SCE). Defaults to 30, so that
        the results are reproducible. See :func:`misc.check_random_state` for further information.

    Returns
    ----------
    ce : float
         The coalition entropy. It is NaN (with a warning) if the signal contains missing values.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute coalition entropy, and, for SCE, the value of each channel (``"Values"``).

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
    elif (isinstance(signal, np.ndarray) and len(signal.shape) == 1) or isinstance(signal, (list, pd.Series)):
        raise ValueError(
            "entropy_coalition(): The input must be a dataframe containing multiple signals.",
        )
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2 or signal.shape[0] < 2:
        raise ValueError("entropy_coalition(): The input must contain at least 2 channels.")

    # Method
    method = method.lower()
    if method in ["ace", "amplitude"]:
        info = {"Method": "ACE"}
    elif method in ["sce", "synchrony"]:
        info = {"Method": "SCE"}
    else:
        raise ValueError("entropy_coalition(): `method` must be 'amplitude' (or 'ACE') or 'synchrony' (or 'SCE').")

    if np.any(np.isnan(signal)):
        warn(
            "The signal contains missing values, so the coalition entropy cannot be computed. Returning NaN.",
            category=NeuroKitWarning,
        )
        if info["Method"] == "SCE":
            info["Values"] = np.full(len(signal), np.nan)
        return np.nan, info

    # Detrend and remove the mean
    signal = np.array([signal_detrend(i - np.mean(i)) for i in signal])

    # Random number generator for the normalization
    rng = check_random_state(random_state)

    if info["Method"] == "ACE":
        entropy = _entropy_coalition_amplitude(signal, rng)
    else:
        entropy, info["Values"] = _entropy_coalition_synchrony(signal, rng)

    return entropy, info


# =============================================================================
# Methods
# =============================================================================


def _entropy_coalition_synchrony(signal, rng):
    n_channels, n_samples = np.shape(signal)

    # Get binary matrices of synchrony for each series
    transformed = np.angle(scipy.signal.hilbert(signal))

    # Create random binary matrix for normalization
    y = rng.random((n_channels - 1, n_samples))
    random_binarized = np.array([_signal_binarize_threshold(i, threshold=0.5) for i in y])
    norm = _entropy_coalition_shannon(random_binarized)

    # Compute shannon entropy of the synchrony series of each channel with the others
    entropy = np.zeros(n_channels)
    for i in range(n_channels):
        others = transformed[np.arange(n_channels) != i]
        entropy[i] = _entropy_coalition_shannon(_entropy_coalition_synchrony_phase(transformed[i], others))

    return np.mean(entropy) / norm, entropy / norm


def _entropy_coalition_amplitude(signal, rng):
    # Hilbert transform to determine the amplitude envelope
    env = np.array([np.abs(scipy.signal.hilbert(i)) for i in signal])

    # Binarize (similar to LZC), mean of absolute of signal as threshold
    binarized = np.array([_signal_binarize_threshold(i, threshold="mean") for i in env])

    # Compute Shannon Entropy
    e1 = _entropy_coalition_shannon(binarized)

    # Shuffle (each channel independently)
    for seq in binarized:
        rng.shuffle(seq)

    # Shuffled result as normalization
    e2 = _entropy_coalition_shannon(binarized)

    return e1 / e2


# =============================================================================
# Utilities
# =============================================================================


def _entropy_coalition_synchrony_phase(phase1, phase2):
    """Compute synchrony of two series of phases (or of one with several, as rows): 1 when their
    difference (on the circle) is less than 0.8 radians"""
    diff = np.abs(phase1 - phase2)
    diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)
    return (diff < 0.8).astype(float)


def _entropy_coalition_shannon(binary_sequence):
    """Shannon entropy of the coalitions, i.e., of the binary columns (time points) of the binary
    matrix (channels x time points)"""
    binary_sequence = np.asarray(binary_sequence, dtype=np.int64)
    if len(binary_sequence) <= 62:
        # Map each binary column onto an integer
        mapped = (2 ** np.arange(len(binary_sequence), dtype=np.int64)) @ binary_sequence
        _, counts = np.unique(mapped, return_counts=True)
    else:
        # Too many channels for the integers: compare the columns themselves
        _, counts = np.unique(binary_sequence, axis=1, return_counts=True)
    return entropy_shannon(freq=counts)[0]
