from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from .utils_complexity_embedding import complexity_embedding


def entropy_correctedconditional(signal, delay=1, dimension=10, c=6, base=2, show=False):
    """**Corrected Conditional Entropy (CCE)**

    The corrected conditional entropy (Porta et al., 1998) estimates the amount of information
    carried by a new sample that cannot be predicted from the previous ones, i.e., the entropy rate
    of the signal, with a correction that makes it usable on short series (e.g., 300 heartbeats).
    It is a standard index of regularity in cardiovascular variability.

    The signal is uniformly quantized into ``c`` levels over its range (6 by default, as in Porta et
    al., 1998), and cut into patterns of *L* successive samples (``delay`` samples apart). The
    conditional entropy of a sample given the *L - 1* previous ones is the difference between the
    Shannon entropies of the patterns of *L* and *L - 1* samples:

    .. math::

      CE(L) = E(L) - E(L - 1)

    with :math:`E(0) = 0` (these are the block entropies of :func:`entropy_rate`, which estimates
    the entropy rate from their slope instead). As *L* increases, the patterns become so many that most of them occur
    only once, which makes them look perfectly predictable: *CE* decreases towards 0 even for white
    noise. The correction adds the entropy of a fully unpredictable sample, :math:`E(1)`, in
    proportion to the patterns found only once, :math:`perc(L)` (Porta et al., 1998):

    .. math::

      CCE(L) = CE(L) + perc(L) \\cdot E(1)

    The CCE is the minimum of :math:`CCE(L)` over *L*, which is the best estimate of the
    conditional entropy given the length of the signal. It is also returned normalized by
    :math:`E(1)` (*NCCE*, between 0 and about 1), together with the regularity index
    :math:`\\rho = 1 - NCCE` (Porta et al., 2001), which is close to 0 for white noise and to 1 for a
    periodic signal.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples,
        between the samples of the patterns.
    dimension : int
        The maximum length *L* of the patterns. The minimum of :math:`CCE(L)` is searched for *L*
        between 1 and ``dimension``. It is usually reached for a small *L* (e.g., 2 to 4), as the
        number of possible patterns, :math:`c^L`, quickly exceeds the length of the signal.
    c : int
        The number of quantization levels (often denoted :math:`\\xi`).
    base : float
        The logarithmic base used for the entropies (default 2, i.e., bits). Use ``np.e`` for nats.
        The normalized CCE and the regularity index do not depend on it.
    show : bool
        If ``True``, plots the conditional entropy, the correction and the CCE as a function of
        the length of the patterns.

    Returns
    -------
    CCE : float
        The corrected conditional entropy (the minimum of :math:`CCE(L)`).
    info : dict
        A dictionary containing additional information, such as the normalized CCE (``"NCCE"``),
        the regularity index (``"Regularity"``), the length of the patterns at the minimum
        (``"L"``), and the values for each length (``"CCE_L"``, ``"CE_L"`` and ``"Perc_L"``, for
        *L* from 1 to ``dimension``).

    See Also
    --------
    entropy_rate, entropy_shannon, entropy_permutation

    Examples
    ----------
    .. ipython:: python

      import numpy as np
      import neurokit2 as nk

      # Short series of about 350 RR intervals
      ecg = nk.ecg_simulate(duration=300, sampling_rate=100, heart_rate=70, random_state=42)
      peaks, info = nk.ecg_peaks(ecg, sampling_rate=100)
      rri = np.diff(info["ECG_R_Peaks"]) / 100 * 1000

      @savefig p_entropy_correctedconditional1.png scale=100%
      cce, info = nk.entropy_correctedconditional(rri, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      cce
      info["NCCE"]
      info["Regularity"]

    References
    ----------
    * Porta, A., Baselli, G., Liberati, D., Montano, N., Cogliati, C., Gnecchi-Ruscone, T.,
      Malliani, A., & Cerutti, S. (1998). Measuring regularity by means of a corrected conditional
      entropy in sympathetic outflow. Biological Cybernetics, 78(1), 71-78.
    * Porta, A., Guzzetti, S., Montano, N., Furlan, R., Pagani, M., Malliani, A., & Cerutti, S.
      (2001). Entropy, entropy rate, and pattern classification as tools to typify complexity in
      short heart period variability series. IEEE Transactions on Biomedical Engineering, 48(11),
      1282-1291.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    if not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError("NeuroKit error: entropy_correctedconditional(): `dimension` must be an integer of at least 1.")
    if not isinstance(c, (int, np.integer)) or c < 2:
        raise ValueError("NeuroKit error: entropy_correctedconditional(): `c` must be an integer of at least 2.")

    signal = np.asarray(signal, dtype=float)
    info = {"Dimension": dimension, "Delay": delay, "c": c}
    undefined = {"NCCE": np.nan, "Regularity": np.nan, "L": np.nan}
    if np.any(np.isnan(signal)):
        warn("The signal contains missing values. Returning NaN.", category=NeuroKitWarning)
        return np.nan, {**info, **undefined}
    if len(signal) == 0 or np.all(signal == signal[0]):
        warn("The signal is constant, its conditional entropy is undefined. Returning NaN.", category=NeuroKitWarning)
        return np.nan, {**info, **undefined}

    # Uniform quantization into c levels over the range of the signal (0 to c - 1)
    symbols = np.floor((signal - np.min(signal)) / np.ptp(signal) * c).astype(int)
    symbols[symbols == c] = c - 1

    # Shannon entropy of the patterns of length L, and proportion of patterns found only once
    entropies = np.zeros(dimension + 1)  # E(0) = 0
    perc = np.zeros(dimension)
    for L in range(1, dimension + 1):
        patterns = complexity_embedding(symbols, dimension=L, delay=delay).astype(np.int64)
        if L * np.log2(c) < 62:
            _, counts = np.unique(patterns @ (c ** np.arange(L, dtype=np.int64)), return_counts=True)
        else:
            _, counts = np.unique(patterns, axis=0, return_counts=True)
        p = counts / counts.sum()
        entropies[L] = -np.sum(p * np.log(p)) / np.log(base)
        perc[L - 1] = np.sum(counts == 1) / counts.sum()

    ce = np.diff(entropies)
    cce = ce + perc * entropies[1]
    imin = np.argmin(cce)
    ncce = cce[imin] / entropies[1]

    info.update(
        {
            "NCCE": ncce,
            "Regularity": 1 - ncce,
            "L": imin + 1,
            "CCE_L": cce,
            "CE_L": ce,
            "Perc_L": perc,
        }
    )

    if show is True:
        lengths = np.arange(1, dimension + 1)
        plt.plot(lengths, ce, "o-", color="#2196F3", label="Conditional entropy CE(L)")
        plt.plot(lengths, perc * entropies[1], "o-", color="#FF9800", label="Correction perc(L) * E(1)")
        plt.plot(lengths, cce, "o-", color="#E91E63", label="Corrected CE(L)")
        plt.axvline(imin + 1, color="#9E9E9E", linestyle="--", label=f"Minimum (L = {imin + 1})")
        plt.xlabel("Length of the patterns (L)")
        plt.ylabel("Entropy" + (" (bits)" if base == 2 else ""))
        plt.title("Corrected Conditional Entropy")
        plt.legend()

    return cce[imin], info
