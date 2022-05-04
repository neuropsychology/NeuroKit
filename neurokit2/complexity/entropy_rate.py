import matplotlib.pyplot as plt
import numpy as np

from .entropy_shannon import entropy_shannon


def entropy_rate(signal, kmax=6, show=False):
    """**Entropy Rate (RatEn)**

    The Entropy Rate (RatEn or ER) quantifies the amount of information needed to describe the
    signal given observations of signal(k). In other words, it is the entropy of the time series
    conditioned on the *k*-histories.

    It quantifies how much uncertainty or randomness the process produces at each new time step,
    given knowledge about the past states of the process. The entropy rate is estimated as the s
    lope of the linear fit between the history length *k* and the joint Shannon entropies. The
    intercept is called **excess entropy**.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        A :func:`symbolic <complexity_symbolize>` sequence in the form of a vector of values.
    kmax : int
        The max history length to consider.
    show : bool
        Plot the Entropy Rate line.

    See Also
    --------
    entropy_shannon

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = [1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 3, 2, 2, 1, 3, 2]

      @savefig p_entropy_rate1.png scale=100%
      raten, info = nk.entropy_rate(signal, kmax=8, show=True)
      @suppress
      plt.close()

    Here we can see that *kmax* is likely to big to provide an accurate estimation of entropy rate.

    .. ipython:: python

      @savefig p_entropy_rate1.png scale=100%
      raten, info = nk.entropy_rate(signal, kmax=3, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      raten
      info["Excess_Entropy"]


    """
    info = {
        "Entropy": [_selfentropy(signal, k + 1) for k in range(kmax)],
        "k": np.arange(1, kmax + 1),
    }

    raten, info["Excess_Entropy"] = np.polyfit(info["k"], info["Entropy"], 1)

    if show:
        plt.figure(figsize=(6, 6))
        plt.plot(info["k"], info["Entropy"], "-sk")
        plt.plot(
            info["k"],
            raten * info["k"] + info["Excess_Entropy"],
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
        plt.legend(loc="lower right")
        plt.xticks(info["k"])
        plt.xlabel("History Length $k$")
        plt.ylabel("Entropy")
        plt.title("Entropy Rate")
        plt.tight_layout()

    return raten, info


def _selfentropy(x, k=3):
    """Shannon's Self joint entropy with k as the length of k-history"""
    n = len(x)
    z = [np.array(x)[i : i + k] for i in range(n - k)]
    _, freq = np.unique(z, return_counts=True, axis=0)
    freq = freq / (n - k)
    return entropy_shannon(freq=freq, base=2)[0]
