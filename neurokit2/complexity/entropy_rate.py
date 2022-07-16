import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import find_knee
from .entropy_shannon import entropy_shannon
from .utils_complexity_embedding import complexity_embedding
from .utils_complexity_symbolize import complexity_symbolize


def entropy_rate(signal, kmax=10, symbolize="mean", show=False):
    """**Entropy Rate (RatEn)**

    The Entropy Rate (RatEn or ER) quantifies the amount of information needed to describe the
    signal given observations of signal(k). In other words, it is the entropy of the time series
    conditioned on the *k*-histories.

    It quantifies how much uncertainty or randomness the process produces at each new time step,
    given knowledge about the past states of the process. The entropy rate is estimated as the
    slope of the linear fit between the history length *k* and the joint Shannon entropies. The
    entropy at k = 1 is called **Excess Entropy** (ExEn).

    We adapted the algorithm to include a knee-point detection (beyond which the self-Entropy
    reaches a plateau), and if it exists, we additionally re-compute the Entropy Rate up until that
    point. This **Maximum Entropy Rate** (MaxRatEn) can be retrieved from the dictionary.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        A :func:`symbolic <complexity_symbolize>` sequence in the form of a vector of values.
    kmax : int
        The max history length to consider. If an integer is passed, will generate a range from 1
        to kmax.
    symbolize : str
        Method to convert a continuous signal input into a symbolic (discrete) signal. By default,
        assigns 0 and 1 to values below and above the mean. Can be ``None`` to skip the process (in
        case the input is already discrete). See :func:`complexity_symbolize` for details.
    show : bool
        Plot the Entropy Rate line.

    See Also
    --------
    entropy_shannon

    Examples
    ----------
    **Example 1**: A simple discrete signal. We have to specify ``symbolize=None`` as the signal is
    already discrete.

    .. ipython:: python

      import neurokit2 as nk

      signal = [1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 3, 2, 2, 1, 3, 2]

      @savefig p_entropy_rate1.png scale=100%
      raten, info = nk.entropy_rate(signal, kmax=10, symbolize=None, show=True)
      @suppress
      plt.close()

    Here we can see that *kmax* is likely to big to provide an accurate estimation of entropy rate.

    .. ipython:: python

      @savefig p_entropy_rate2.png scale=100%
      raten, info = nk.entropy_rate(signal, kmax=3, symbolize=None, show=True)
      @suppress
      plt.close()

    **Example 2**: A continuous signal.

    .. ipython:: python

      signal = nk.signal_simulate(duration=2, frequency=[5, 12, 40, 60], sampling_rate=1000)

      @savefig p_entropy_rate3.png scale=100%
      raten, info = nk.entropy_rate(signal, kmax=60, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      raten
      info["Excess_Entropy"]
      info["MaxRatEn"]

    References
    ----------
    * Mediano, P. A., Rosas, F. E., Timmermann, C., Roseman, L., Nutt, D. J., Feilding, A., ... &
      Carhart-Harris, R. L. (2020). Effects of external stimulation on psychedelic state
      neurodynamics. Biorxiv.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Force to array
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    # Make discrete
    if np.isscalar(signal) is False:
        signal = complexity_symbolize(signal, method=symbolize)

    # Convert into range if integer
    if np.isscalar(kmax) is True:
        kmax = np.arange(1, kmax + 1)

    # Compute self-entropy
    info = {
        "Entropy": [_selfentropy(signal, k) for k in kmax],
        "k": kmax,
    }

    # Traditional Entropy Rate (on all the values)
    raten, intercept1 = np.polyfit(info["k"], info["Entropy"], 1)

    # Excess Entropy
    info["Excess_Entropy"] = info["Entropy"][0]

    # Max Entropy Rate
    # Detect knee
    try:
        knee = find_knee(info["Entropy"], verbose=False)
    except ValueError:
        knee = len(info["k"]) - 1

    if knee == len(info["k"]) - 1:
        info["MaxRatEn"], intercept2 = raten, np.nan
    else:
        info["MaxRatEn"], intercept2 = np.polyfit(info["k"][0:knee], info["Entropy"][0:knee], 1)

    # Store knee
    info["Knee"] = knee

    # Plot
    if show:
        plt.figure(figsize=(6, 6))
        plt.plot(info["k"], info["Entropy"], "o-", color="black")
        y = raten * info["k"] + intercept1
        plt.plot(
            info["k"],
            y,
            color="red",
            label=f"Entropy Rate = {raten:.2f}",
        )
        plt.plot(
            (np.min(info["k"]), np.min(info["k"])),
            (0, info["Entropy"][0]),
            "--",
            color="blue",
            label=f"Excess Entropy = {info['Excess_Entropy']:.2f}",
        )
        if not np.isnan(intercept2):
            y2 = info["MaxRatEn"] * info["k"] + intercept2
            plt.plot(
                info["k"][y2 <= np.max(y)],
                y2[y2 <= np.max(y)],
                color="purple",
                label=f"Max Entropy Rate = {info['MaxRatEn']:.2f}",
            )
            plt.plot(
                (info["k"][knee], info["k"][knee]),
                (0, info["Entropy"][knee]),
                "--",
                color="purple",
                label=f"Knee  = {info['k'][knee]}",
            )
        plt.legend(loc="lower right")
        plt.xlabel("History Length $k$")
        plt.ylabel("Entropy")
        plt.title("Entropy Rate")

    return raten, info


def _selfentropy(x, k=3):
    """Shannon's Self joint entropy with k as the length of k-history"""
    z = complexity_embedding(x, dimension=int(k), delay=1)
    _, freq = np.unique(z, return_counts=True, axis=0)
    freq = freq / freq.sum()
    return entropy_shannon(freq=freq, base=2)[0]
