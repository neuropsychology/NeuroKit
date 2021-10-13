from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning, find_plateau


def complexity_k(signal, k_max="max", show=False, **kwargs):
    """Automated selection of the optimal k_max parameter for Higuchi Fractal Dimension (HFD).

    The optimal kmax is computed based on the point at which HFD values plateau for a range of kmax values (see Vega, 2015).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    k_max : Union[int, str, list], optional
        Maximum number of interval times (should be greater than or equal to 3) to be tested. If 'max', it selects the maximum possible value corresponding to half the length of the signal.
    show : bool
        Visualise the slope of the curve for the selected kmax value.

    Returns
    --------
    k : float
        The optimal kmax of the time series.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute optimal kmax.

    See Also
    --------
    fractal_higuchi

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=1, sampling_rate=100, frequency=[5, 6], noise=0.5)
    >>> k_max, info = nk.complexity_k(signal, k_max='default', show=True)
    >>> k_max #doctest: +SKIP

    Reference
    ----------
    - Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory.
    Physica D: Nonlinear Phenomena, 31(2), 277-283.

    - Vega, C. F., & Noel, J. (2015, June). Parameters analyzed of Higuchi's fractal dimension for EEG brain signals.
    In 2015 Signal Processing Symposium (SPSympo) (pp. 1-5). IEEE. https://ieeexplore.ieee.org/document/7168285
    """
    # Get the range of k-max values to be tested
    # ------------------------------------------
    if isinstance(k_max, str):  # e.g., "default"
        # upper limit for k value (max possible value)
        k_max = int(np.floor(len(signal) / 2))  # so that normalizing factor is positive

    if isinstance(k_max, int):
        k_range = np.arange(2, k_max + 1)
    elif isinstance(k_max, (list, np.ndarray, pd.Series)):
        k_range = np.array(k_max)
    else:
        warn(
            "k_max should be an int or a list of values of kmax to be tested.",
            category=NeuroKitWarning,
        )

    # Compute the slope for each kmax value
    # --------------------------------------
    slopes = np.zeros(len(k_range))
    intercepts = np.zeros(len(k_range))
    average_values = list(np.zeros(len(k_range)))
    for i, k in enumerate(k_range):
        slopes[i], intercepts[i], _, average_values[i] = _complexity_k_slope(signal, k, **kwargs)

    # Find plateau (the saturation point of slope)
    # --------------------------------------------
    try:
        optimal_point = find_plateau(slopes, show=False)
        k_optimal = k_range[optimal_point]
    except ValueError:  # if can't find plateau
        k_optimal = np.max(k_range)
        warn(
            f"The optimal kmax value detected is 2 or less. There may be no plateau in this case. You can inspect the plot by set `show=True`. We will return optimal k_max = {k_optimal} (the max).",
            category=NeuroKitWarning,
        )

    # if len(k_indices) == 0:
    #     k_optimal = np.max(k_range)
    #     warn(
    #         f"The optimal kmax value detected is 2 or less. There may be no plateau in this case. You can inspect the plot by set `show=True`. We will return optimal k_max = {k_optimal} (the max).",
    #         category=NeuroKitWarning,
    #     )
    # else:
    #     k_optimal = k_range[k_indices[0]]

    # Plot
    if show:
        _complexity_k_plot(k_range, slopes, k_optimal, ax=None)

    # Return optimal tau and info dict
    return k_optimal, {
        "Values": k_range,
        "Scores": slopes,
        "Intercepts": intercepts,
        "Average_Values": average_values,
    }


# =============================================================================
# Utilities
# =============================================================================


def _complexity_k_slope(signal, k, k_number="range"):
    if k_number == "range":
        k_values = np.arange(1, k + 1)
    else:
        k_values = np.unique(np.linspace(1, k + 1, k_number).astype(int))
    average_values = _complexity_k_average_values(signal, k_values)

    # Slope of best-fit line through points
    slope, intercept = -np.polyfit(np.log(k_values), np.log(average_values), 1)
    return slope, intercept, k_values, average_values


def _complexity_k_average_values(signal, k_values):
    """Step 3 of Vega & Noel (2015)"""
    n = len(signal)
    average_values = np.zeros(len(k_values))

    # Compute length of the curve, Lm(k)
    for i, k in enumerate(k_values):
        sets = np.zeros(k)
        for j, m in enumerate(range(1, k + 1)):
            n_max = int(np.floor((n - m) / k))
            normalization = (n - 1) / (n_max * k)
            Lm_k = np.sum(np.abs(np.diff(signal[m - 1 :: k], n=1))) * normalization
            sets[j] = Lm_k / k
        # Compute average value over k sets of Lm(k)
        L_k = np.sum(sets) / k
        average_values[i] = L_k
    return average_values


def _complexity_k_plot(k_range, slope_values, k_optimal, ax=None):

    # Prepare plot
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    ax.set_title("Optimization of $k_{max}$ parameter")
    ax.set_xlabel("$k_{max}$ values")
    ax.set_ylabel("Higuchi Fractal Dimension (HFD) values")
    colors = plt.cm.PuBu(np.linspace(0, 1, len(k_range)))

    # if single time series
    ax.plot(k_range, slope_values, color="#2196F3", zorder=1)
    for i, j in enumerate(k_range):
        ax.scatter(k_range[i], slope_values[i], color=colors[i], marker="o", zorder=2)
    ax.axvline(x=k_optimal, color="#E91E63", label="Optimal $k_{max}$: " + str(k_optimal))
    ax.legend(loc="upper right")

    return fig
