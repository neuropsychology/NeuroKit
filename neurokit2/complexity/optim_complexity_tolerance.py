# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import scipy.spatial

from ..stats import density
from .complexity_embedding import complexity_embedding
from .entropy_approximate import entropy_approximate


def complexity_tolerance(
    signal, method="maxApEn", r_range=None, delay=None, dimension=None, show=False
):
    """**Automated selection of tolerance (r)**

    Estimate and select the optimal tolerance (*r*) parameter used by other entropy and other
    complexity algorithms.

    Many complexity algorithms are built on the notion of self-similarity and recurrence, and how
    often a system revisits its past states. Considering two states as identical is straightforward
    for discrete systems (e.g., a sequence of "A", "B" and "C" states), but for continuous signals,
    we cannot simply look for when the two numbers are exactly the same. Instead, we have to pick a
    threshold by which to consider two points as similar.

    The tolerance *r* is essentially this threshold value (the numerical difference between two
    similar points that we "tolerate"). This parameter has a critical impact and is a major
    source of inconsistencies in the literature.

    Different methods have been described to estimate the most appropriate tolerance value:

    * ``'sd'`` (as in Standard Deviation): r = 0.2 * standard deviation of the signal will be
      returned.
    * ``'maxApEn'``: Different values of tolerance will be tested and the one where the approximate
      entropy (ApEn) is maximized will be selected and returned.
    * ``'recurrence'``, the tolerance that yields a recurrence rate (see ``RQA``) close to 5% will
      be returned.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        Can be 'maxApEn' (default), 'sd', or 'recurrence'.
    r_range : Union[list, int]
        The range of tolerance values (or the number of values) to test. Only used if ``method`` is
        ``'maxApEn'`` or ``'recurrence'``. If ``None`` (default), the default range will be used;
        ``np.linspace(0.02, 0.8, r_range) * np.std(signal, ddof=1)`` for ``'maxApEn'``, and ``np.
        linspace(0, np.max(d), 30 + 1)[1:]`` for ``'recurrence'``. You can set a lower number for
        faster results.
    delay : int
        Only used if ``method='maxApEn'``. See ``entropy_approximate()``.
    dimension : int
        Only used if ``method='maxApEn'``. See ``entropy_approximate()``.
    show : bool
        If true and method is 'maxApEn', will plot the ApEn values for each value of r.

    See Also
    --------
    complexity, complexity_delay, complexity_dimension, complexity_embedding

    Returns
    ----------
    float
        The optimal tolerance value.
    dict
        A dictionary with the values of r and the corresponding ApEn values (when method='maxApEn').

    Examples
    ----------
    * **Example 1**: The method based on the SD of the signal is fast. The plot shows the d
      distribution of the values making the signal, and the width of the arrow represents the
      chosen ``r`` parameter.

    .. ipython:: python

      import neurokit2 as nk

      # Simulate signal
      signal = nk.signal_simulate(duration=2, frequency=5)

      # Fast method (based on the standard deviation)
      @savefig p_complexity_tolerance1.png scale=100%
      r, info = nk.complexity_tolerance(signal, method = 'SD', show=True)
      r
      @suppress
      plt.close()

    * **Example 2**: The method based on the recurrence rate will display the rates according to
      different values of tolerance. The horizontal line indicates 5%.

    .. ipython:: python

      @savefig p_complexity_tolerance2.png scale=100%
      r, info = nk.complexity_tolerance(signal, delay=1, dimension=10,
                                        method = 'recurrence', show=True)
      r
      @suppress
      plt.close()

    * **Example 3**: The default method selects the tolerance at which *ApEn* is maximized.

    .. ipython:: python

      # Slow method
      @savefig p_complexity_tolerance3.png scale=100%
      r, info = nk.complexity_tolerance(signal, delay=8, dimension=6,
                                        method = 'maxApEn', show=True)
      r
      @suppress
      plt.close()

    * **Example 4**: The tolerance values that are tested can be modified to get a more precise
      estimate.

    .. ipython:: python

      # Narrower range
      @savefig p_complexity_tolerance4.png scale=100%
      r, info = nk.complexity_tolerance(signal, delay=8, dimension=6, method = 'maxApEn',
                                        r_range=np.linspace(0.002, 0.8, 30), show=True)
      r
      @suppress
      plt.close()

    References
    -----------
    * Lu, S., Chen, X., Kanters, J. K., Solomon, I. C., & Chon, K. H. (2008). Automatic selection of
      the threshold value r for approximate entropy. IEEE Transactions on Biomedical Engineering,
      55(8), 1966-1972.
    """
    if not isinstance(method, str):
        return method, {"Method": "None"}

    # Method
    method = method.lower()
    if method in ["traditional", "sd", "std", "default"]:
        r = 0.2 * np.std(signal, ddof=1)
        info = {"Method": "20% SD"}
    elif method in ["maxapen", "optimize"]:
        r, info = _optimize_tolerance_maxapen(
            signal, r_range=r_range, delay=delay, dimension=dimension
        )
        info.update({"Method": "Max ApEn"})
    elif method in ["recurrence", "rqa"]:
        r, info = _optimize_tolerance_recurrence(
            signal, r_range=r_range, delay=delay, dimension=dimension
        )
        info.update({"Method": "5% Recurrence Rate"})
    else:
        raise ValueError("NeuroKit error: complexity_tolerance(): 'method' not recognized.")

    if show is True:
        _optimize_tolerance_plot(r, info, method=method, signal=signal)
    return r, info


# =============================================================================
# Internals
# =============================================================================
def _optimize_tolerance_recurrence(signal, r_range=None, delay=None, dimension=None):

    # Optimize missing parameters
    if delay is None or dimension is None:
        raise ValueError("If method='RQA', both delay and dimension must be specified.")

    # Compute distance matrix
    emb = complexity_embedding(signal, delay=delay, dimension=dimension)
    d = scipy.spatial.distance.cdist(emb, emb, metric="euclidean")

    if r_range is None:
        r_range = 50
    if isinstance(r_range, int):
        r_range = np.linspace(0, np.max(d), r_range + 1)[1:]

    recurrence_rate = np.zeros_like(r_range)
    # Indices of the lower triangular (without the diagonal)
    idx = np.tril_indices(len(d), k=-1)
    n = len(d[idx])
    for i, r in enumerate(r_range):
        recurrence_rate[i] = (d[idx] <= r).sum() / n
    # Closest to 0.05 (5%)
    optimal = r_range[np.abs(recurrence_rate - 0.05).argmin()]

    return optimal, {"Values": r_range, "Scores": recurrence_rate}


def _optimize_tolerance_maxapen(signal, r_range=None, delay=None, dimension=None):

    # Optimize missing parameters
    if delay is None or dimension is None:
        raise ValueError("If method='maxApEn', both delay and dimension must be specified.")

    if r_range is None:
        r_range = 40
    if isinstance(r_range, int):
        r_range = np.linspace(0.02, 0.8, r_range) * np.std(signal, ddof=1)

    ApEn = np.zeros_like(r_range)
    for i, r in enumerate(r_range):
        ApEn[i], _ = entropy_approximate(
            signal, delay=delay, dimension=dimension, tolerance=r_range[i]
        )

    return r_range[np.argmax(ApEn)], {"Values": r_range, "Scores": ApEn}


# =============================================================================
# Plotting
# =============================================================================
def _optimize_tolerance_plot(r, info, ax=None, method="maxApEn", signal=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if method in ["traditional", "sd", "std", "default", "none"]:
        fig, ax = plt.subplots()
        x, y = density(signal)
        arrow_y = np.mean([np.max(y), np.min(y)])
        x_range = np.max(x) - np.min(x)
        ax.plot(x, y, color="#80059c", label="Optimal r: " + str(np.round(r, 3)))
        ax.arrow(
            np.mean(x),
            arrow_y,
            np.mean(x) + r / 2,
            0,
            head_width=0.01 * x_range,
            head_length=0.01 * x_range,
            linewidth=4,
            color="g",
            length_includes_head=True,
        )
        ax.arrow(
            np.mean(x),
            arrow_y,
            np.mean(x) - r / 2,
            0,
            head_width=0.01 * x_range,
            head_length=0.01 * x_range,
            linewidth=4,
            color="g",
            length_includes_head=True,
        )
        ax.set_title("Optimization of Tolerance Threshold (r)")
        ax.set_xlabel("Signal values")
        ax.set_ylabel("Distribution")
        ax.legend(loc="upper right")
        return fig

    r_range = info["Values"]
    y_values = info["Scores"]

    # Custom legend depending on method
    if method in ["maxapen", "optimize"]:
        ylabel = "Approximate Entropy $ApEn$"
        legend = "$ApEn$"
    else:
        ylabel = "Recurrence Rate $RR$"
        legend = "$RR$"
        y_values *= 100  # Convert to percentage
        ax.axhline(y=0.5, color="grey")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())

    ax.set_title("Optimization of Tolerance Threshold (r)")
    ax.set_xlabel("Tolerance threshold $r$")
    ax.set_ylabel(ylabel)
    ax.plot(r_range, y_values, "o-", label=legend, color="#80059c")
    ax.axvline(x=r, color="#E91E63", label="Optimal r: " + str(np.round(r, 3)))
    ax.legend(loc="upper right")

    return fig
