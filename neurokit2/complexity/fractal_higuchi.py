from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from .optim_complexity_k import _complexity_k_Lk, _complexity_k_slope, complexity_k


def fractal_higuchi(signal, k_max="default", show=False, **kwargs):
    """**Higuchi's Fractal Dimension (HFD)**

    The Higuchi's Fractal Dimension (HFD) is an approximate value for the box-counting dimension for
    time series. It is computed by reconstructing k-max number of new data sets. For each
    reconstructed data set, curve length is computed and plotted against its corresponding
    *k*-value on a log-log scale. HFD corresponds to the slope of the least-squares linear trend.

    Values should fall between 1 and 2. For more information about the *k* parameter selection, see
    the :func:`complexity_k` optimization function.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    k_max : str or int
        Maximum number of interval times, between 2 and half the length of the signal (otherwise,
        HFD is returned as NaN). If ``"default"``, the optimal k-max is estimated using
        :func:`complexity_k`.
    show : bool
        Visualise the slope of the curve for the selected k_max value.
    **kwargs : optional
        Currently not used.

    Returns
    ----------
    HFD : float
        Higuchi's fractal dimension of the time series.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute Higuchi's fractal dimension.

    See Also
    --------
    complexity_k

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, sampling_rate=100, frequency=[3, 6], noise=0.2)

      @savefig p_fractal_higuchi1.png scale=100%
      k_max, info =  nk.complexity_k(signal, k_max='default', show=True)
      @suppress
      plt.close()

      @savefig p_fractal_higuchi2.png scale=100%
      hfd, info = nk.fractal_higuchi(signal, k_max=k_max, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      hfd

    References
    ----------
    * Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory.
      Physica D: Nonlinear Phenomena, 31(2), 277-283.
    * Vega, C. F., & Noel, J. (2015, June). Parameters analyzed of Higuchi's fractal dimension for
      EEG brain signals. In 2015 Signal Processing Symposium (SPSympo) (pp. 1-5). IEEE.
      https://ieeexplore.ieee.org/document/7168285

    """

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet.")
    signal = np.asarray(signal, dtype=float)

    # Get k_max
    if isinstance(k_max, (str, list, np.ndarray, pd.Series)):
        # Optimizing needed
        k_max, info = complexity_k(signal, k_max=k_max, show=False)
        idx = np.where(info["Values"] == k_max)[0][0]
        slope = info["Scores"][idx]
        intercept = info["Intercepts"][idx]
        average_values = info["Average_Values"][idx]
        k_values = np.arange(1, k_max + 1)
    elif not 2 <= k_max <= len(signal) // 2:
        warn(
            f"k_max should be between 2 and half the length of the signal ({len(signal) // 2}). "
            + "HFD cannot be computed and is returned as NaN.",
            category=NeuroKitWarning,
        )
        return np.nan, {"k_max": k_max, "Values": np.array([]), "Scores": np.array([]), "Intercept": np.nan}
    else:
        # Compute Higuchi
        k_values = np.arange(1, k_max + 1)
        average_values = _complexity_k_Lk(signal, k_max)
        slope, intercept = _complexity_k_slope(average_values)
        slope, intercept = slope[-1], intercept[-1]

    # Plot
    if show:
        _fractal_higuchi_plot(k_values, average_values, k_max, slope, intercept)

    return slope, {
        "k_max": k_max,
        "Values": k_values,
        "Scores": average_values,
        "Intercept": intercept,
    }


# =============================================================================
# Utilities
# =============================================================================


def _fractal_higuchi_plot(k_values, average_values, kmax, slope, intercept, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle("Higuchi Fractal Dimension (HFD)")
    else:
        fig = None

    ax.set_title("Least-squares linear best-fit curve for $k_{max}$ = " + str(kmax) + ", slope = " + str(np.round(slope, 2)))
    ax.set_ylabel(r"$ln$(L(k))")
    ax.set_xlabel(r"$ln$(1/k)")
    colors = plt.cm.plasma(np.linspace(0, 1, len(k_values)))

    # Label all values unless len(k_values) > 10 then label only min and max k_max
    if len(k_values) < 10:
        for i in range(0, len(k_values)):
            ax.scatter(
                -np.log(k_values[i]),
                np.log(average_values[i]),
                color=colors[i],
                marker="o",
                zorder=2,
                label=f"k = {i + 1}",
            )
    else:
        for i in range(0, len(k_values)):
            ax.scatter(
                -np.log(k_values[i]),
                np.log(average_values[i]),
                color=colors[i],
                marker="o",
                zorder=2,
                label="_no_legend_",
            )
        ax.plot([], label=f"k = {np.min(k_values)}", c=colors[0])
        ax.plot([], label=f"k = {np.max(k_values)}", c=colors[-1])

    fit_values = slope * -np.log(k_values) + intercept
    ax.plot(-np.log(k_values), fit_values, color="#FF9800", zorder=1)
    ax.legend(loc="lower right")

    return fig
