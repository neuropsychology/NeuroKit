from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning


def complexity_k(signal, k_max="max", show=False):
    """**Automated selection of k for Higuchi Fractal Dimension (HFD)**

    The optimal *k-max* is computed based on the point at which HFD values plateau for a range of
    *k-max* values (see Vega & Noel, 2015): it is the first *k-max* from which the HFD stays within
    0.01 of its value for the next 10 tested *k-max* values. If there is no such point (e.g., when
    fewer than 11 values are tested), the largest *k-max* tested is returned with a warning.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    k_max : Union[int, str, list], optional
        Maximum number of interval times (should be greater than or equal to 2) to be tested. If
        a string (e.g., ``"max"``), it tests all the values up to the maximum possible one, half
        the length of the signal. If a list, it tests these values in increasing order (between 2
        and half the length of the signal).
    show : bool
        Visualise the slope of the curve for the selected kmax value.

    Returns
    --------
    k : int
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
    # Sanity checks
    signal = np.asarray(signal, dtype=float)
    if signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")

    # Get the range of k-max values to be tested
    # ------------------------------------------
    if isinstance(k_max, str):  # e.g., "default"
        # upper limit for k value (max possible value)
        k_max = len(signal) // 2  # so that normalizing factor is positive

    if isinstance(k_max, (int, np.integer)):
        kmax_range = np.arange(2, k_max + 1)
    elif isinstance(k_max, (list, np.ndarray, pd.Series)):
        kmax_range = np.unique(np.asarray(k_max, dtype=int))
    else:
        raise ValueError("k_max should be a string, an int or a list of values of kmax to be tested.")

    if len(kmax_range) == 0 or np.min(kmax_range) < 2 or np.max(kmax_range) > len(signal) // 2:
        raise ValueError(
            f"The k_max values to be tested should be between 2 and half the length of the signal ({len(signal) // 2})."
        )

    # Compute the slope for each kmax value
    # --------------------------------------
    # The curve lengths L(k) do not depend on kmax, so they are computed once up to the largest one
    average_values = _complexity_k_Lk(signal, np.max(kmax_range))
    slopes, intercepts = _complexity_k_slope(average_values)
    slopes, intercepts = slopes[kmax_range - 1], intercepts[kmax_range - 1]
    average_values = [average_values[:k] for k in kmax_range]

    # Find plateau (the saturation point of slope)
    # --------------------------------------------
    optimal_point = _complexity_k_plateau(slopes)
    if optimal_point is not None:
        kmax_optimal = int(kmax_range[optimal_point])
    else:
        kmax_optimal = int(np.max(kmax_range))
        warn(
            "No plateau was found in the HFD values. You can inspect the plot by setting `show=True`. "
            + f"We will return the largest k_max tested ({kmax_optimal}).",
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


def _complexity_k_Lk(signal, k_max):
    """Average length L(k) of the curve for k = 1, ..., k_max (Higuchi, 1988)."""
    n = len(signal)
    average_values = np.zeros(k_max)
    for k in range(1, k_max + 1):
        # Step 1: the increments |x(i + k) - x(i)| of the k new time series x(m), x(m + k), ...,
        # where the increment starting at i belongs to the series m = i % k (0-indexed)
        increments = np.abs(signal[k:] - signal[:-k])
        lengths = np.bincount(np.arange(n - k) % k, weights=increments, minlength=k)

        # Step 2: Calculate length Lm(k) of each curve
        n_increments = (n - np.arange(1, k + 1)) // k  # floor((n - m) / k), where m = 1, 2... k
        lengths = lengths * (n - 1) / (n_increments * k) / k

        # Step 3: Compute average value over k sets of Lm(k)
        average_values[k - 1] = np.mean(lengths)
    return average_values


def _complexity_k_slope(average_values):
    """HFD (slope) and intercept of log L(k) against log(1/k), for k_max = 1, ..., len(average_values)."""
    # Least-squares fits of every k = 1, ..., k_max, all at once from cumulative sums
    x = np.log(1 / np.arange(1, len(average_values) + 1))
    y = np.log(average_values)
    n = np.arange(1, len(x) + 1)
    sum_x, sum_y = np.cumsum(x), np.cumsum(y)
    with np.errstate(divide="ignore", invalid="ignore"):  # a single point (k_max = 1) has no slope
        slopes = (n * np.cumsum(x * y) - sum_x * sum_y) / (n * np.cumsum(x**2) - sum_x**2)
    intercepts = (sum_y - slopes * sum_x) / n
    return slopes, intercepts


def _complexity_k_plateau(slopes, tolerance=0.01, window=10):
    """Index of the first HFD value from which the next ``window`` values stay within ``tolerance``."""
    if len(slopes) <= window:
        return None
    following = np.lib.stride_tricks.sliding_window_view(slopes[1:], window)
    stable = np.all(np.abs(following - slopes[: len(following), None]) < tolerance, axis=1)
    return int(np.argmax(stable)) if np.any(stable) else None


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
