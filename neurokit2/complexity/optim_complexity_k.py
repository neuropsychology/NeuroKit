from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning, find_plateau


def complexity_k(signal, k_max="max", show=False):
    """**Automated selection of k for Higuchi Fractal Dimension (HFD)**

    The optimal *k-max* is computed based on the point at which HFD values plateau for a range of
    *k-max* values (see Vega, 2015).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    k_max : Union[int, str, list], optional
        Maximum number of interval times (should be greater than or equal to 3) to be tested. If
        ``max``, it selects the maximum possible value corresponding to half the length of the
        signal.
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
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, sampling_rate=100, frequency=[5, 6], noise=0.5)

      @savefig p_complexity_k1.png scale=100%
      k_max, info = nk.complexity_k(signal, k_max='default', show=True)
      @suppress
      plt.close()

    .. ipython:: python

      k_max

    References
    ----------
    * Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory.
      Physica D: Nonlinear Phenomena, 31(2), 277-283.
    * Vega, C. F., & Noel, J. (2015, June). Parameters analyzed of Higuchi's fractal dimension for
      EEG brain signals. In 2015 Signal Processing Symposium (SPSympo) (pp. 1-5). IEEE. https://
      ieeexplore.ieee.org/document/7168285

    """
    # Get the range of k-max values to be tested
    # ------------------------------------------
    if isinstance(k_max, str):  # e.g., "default"
        # upper limit for k value (max possible value)
        k_max = int(np.floor(len(signal) / 2))  # so that normalizing factor is positive

    if isinstance(k_max, int):
        kmax_range = np.arange(2, k_max + 1)
    elif isinstance(k_max, (list, np.ndarray, pd.Series)):
        kmax_range = np.array(k_max)
    else:
        warn(
            "k_max should be an int or a list of values of kmax to be tested.",
            category=NeuroKitWarning,
        )

    # Compute the slope for each kmax value
    # --------------------------------------
    vectorized_k_slope = np.vectorize(_complexity_k_slope, excluded=[1])
    slopes, intercepts, info = vectorized_k_slope(kmax_range, signal)
    # k_values = [d["k_values"] for d in info]
    average_values = [d["average_values"] for d in info]

    # Find plateau (the saturation point of slope)
    # --------------------------------------------
    optimal_point = find_plateau(slopes, show=False)
    if optimal_point is not None:
        kmax_optimal = kmax_range[optimal_point]
    else:
        kmax_optimal = np.max(kmax_range)
        warn(
            "The optimal kmax value detected is 2 or less. There may be no plateau in this case. "
            + f"You can inspect the plot by set `show=True`. We will return optimal k_max = {kmax_optimal} (the max).",
            category=NeuroKitWarning,
        )

    # Plot
    if show:
        _complexity_k_plot(kmax_range, slopes, kmax_optimal, ax=None)

    # Return optimal tau and info dict
    return kmax_optimal, {
        "Values": kmax_range,
        "Scores": slopes,
        "Intercepts": intercepts,
        "Average_Values": average_values,
    }


# =============================================================================
# Utilities
# =============================================================================


def _complexity_k_Lk(k, signal):
    n = len(signal)

    # Step 1: construct k number of new time series for range of k_values from 1 to kmax
    k_subrange = np.arange(1, k + 1)  # where m = 1, 2... k

    idx = np.tile(np.arange(0, len(signal), k), (k, 1)).astype(float)
    idx += np.tile(np.arange(0, k), (idx.shape[1], 1)).T
    mask = idx >= len(signal)
    idx[mask] = 0

    sig_values = signal[idx.astype(int)].astype(float)
    sig_values[mask] = np.nan

    # Step 2: Calculate length Lm(k) of each curve
    normalization = (n - 1) / (np.floor((n - k_subrange) / k).astype(int) * k)
    sets = (np.nansum(np.abs(np.diff(sig_values)), axis=1) * normalization) / k

    # Step 3: Compute average value over k sets of Lm(k)
    return np.sum(sets) / k


def _complexity_k_slope(kmax, signal, k_number="max"):
    if k_number == "max":
        k_values = np.arange(1, kmax + 1)
    else:
        k_values = np.unique(np.linspace(1, kmax + 1, k_number).astype(int))

    # Step 3 of Vega & Noel (2015)
    vectorized_Lk = np.vectorize(_complexity_k_Lk, excluded=[1])

    # Compute length of the curve, Lm(k)
    average_values = vectorized_Lk(k_values, signal)

    # Slope of best-fit line through points (slope equal to FD)
    slope, intercept = -np.polyfit(np.log(k_values), np.log(average_values), 1)
    return slope, intercept, {"k_values": k_values, "average_values": average_values}


# =============================================================================
# Plotting
# =============================================================================


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
    for i, _ in enumerate(k_range):
        ax.scatter(k_range[i], slope_values[i], color=colors[i], marker="o", zorder=2)
    ax.axvline(x=k_optimal, color="#E91E63", label="Optimal $k_{max}$: " + str(k_optimal))
    ax.legend(loc="upper right")

    return fig
