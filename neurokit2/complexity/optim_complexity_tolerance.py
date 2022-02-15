# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import scipy.spatial

from .complexity_embedding import complexity_embedding
from .entropy_approximate import entropy_approximate


def complexity_tolerance(
    signal, method="maxApEn", r_range=None, delay=None, dimension=None, show=False
):
    """Automated selection of the optimal tolerance (r) parameter for entropy measures

    The tolerance r is essentially a threshold value by which to consider two points as similar.
    This parameter has a critical impact and is a major source of inconsistencies in the literature.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        If 'maxApEn', different values of tolerance will be tested and the one where ApEn is
        maximized will be selected and returned. If 'sd' (as in Standard Deviation),
        r = 0.2 * standard deviation of the signal will be returned. If 'recurrence', the tolerance that
        yields a recurrence rate (see ``RQA``) close to 5% will be returned.
    r_range : Union[list, int]
        Only used if ``method='maxApEn'``. The range of tolerance values to test.
        If an integer, will be set to ``np.linspace(0.02, 0.8, r_range) * np.std(signal, ddof=1)``.
        If ``None``, will be set to ``40``. You can set a lower number for faster results.
    delay : int
        Only used if ``method='maxApEn'``. See ``entropy_approximate()``.
    dimension : int
        Only used if ``method='maxApEn'``. See ``entropy_approximate()``.
    show : bool
        If true and method is 'maxApEn', will plot the ApEn values for each value of r.

    See Also
    --------
    entropy_approximate, complexity_delay, complexity_dimension

    Returns
    ----------
    float
        The optimal tolerance value.
    dict
        A dictionary with the values of r and the corresponding ApEn values (when method='maxApEn').

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>>
    >>> # Fast
    >>> r, info = nk.complexity_tolerance(signal, method = 'SD')
    >>> r
    0.07072836242007384
    >>>
    >>> r, info = nk.complexity_tolerance(signal, delay=1, dimension=10, method = 'recurrence', show=True)
    >>> r
    0.06298107683978625
    >>>
    >>> # Slow
    >>> r, info = nk.complexity_tolerance(signal, delay=8, dimension=6, method = 'maxApEn', show=True)
    >>> r
    0.014145672484014769
    >>>
    >>> # Narrower range
    >>> r, info = nk.complexity_tolerance(signal, delay=8, dimension=6, method = 'maxApEn',
    ...                                   r_range=np.linspace(0.002, 0.1, 30), show=True)
    >>>


    References
    -----------
    - Lu, S., Chen, X., Kanters, J. K., Solomon, I. C., & Chon, K. H. (2008). Automatic selection of
      the threshold value r for approximate entropy. IEEE Transactions on Biomedical Engineering,
      55(8), 1966-1972.
    """
    # Method
    method = method.lower()
    if method in ["traditional", "sd", "std"]:
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
        _optimize_tolerance_plot(r, info["Values"], info["Scores"], method=method)
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
    optimal = r_range[np.abs(r_range - 0.05).argmin()]

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
def _optimize_tolerance_plot(r, r_range, y_values, ax=None, method="maxApEn"):
    if method in ["traditional", "sd", "std"]:
        return None

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    # Custom legend depending on method
    if method in ["maxapen", "optimize"]:
        ylabel = "Approximate Entropy $ApEn$"
        legend = "$ApEn$"
    else:
        ylabel = "Recurrence Rate $RR$"
        legend = "$RR$"
        y_values *= 100  # Convert to percentage
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())

    ax.set_title("Optimization of Tolerance Threshold (r)")
    ax.set_xlabel("Tolerance threshold $r$")
    ax.set_ylabel(ylabel)
    ax.plot(r_range, y_values, "o-", label=legend, color="#80059c")
    ax.axvline(x=r, color="#E91E63", label="Optimal r: " + str(np.round(r, 3)))
    ax.legend(loc="upper right")

    return fig
