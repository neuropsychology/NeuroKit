# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from .complexity_k import _complexity_k_slope, complexity_k, _complexity_k_plot
from .utils import _sanitize_multichannel


def fractal_higuchi(signal, k_max="default", show=False):
    """
    Computes Higuchi's Fractal Dimension (HFD) by reconstructing k-max number of new
    data sets. For each reconstructed data set, curve length is computed and plotted
    against its corresponding k value on a log-log scale. HFD equates to the slope obtained
    from fitting a least-squares method.

    Values should fall between 1 and 2. For more information about k parameter selection, see
    the ``complexity_k()`` optimization function.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    k_max : str or int
        Maximum number of interval times (should be greater than or equal to 2).
        If "default", then the optimal kmax is computed based on the point at which HFD values plateau
        for a range of kmax values (see ``complexity_k()`` optimization function).
    show : bool
        Visualise the slope of the curve for the selected k_max value.

    Returns
    ----------
    slope : float
        Higuchi's fractal dimension of the single time series.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute Higuchi's fractal dimension.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=1, sampling_rate=100, frequency=[3, 6], noise = 0.2)
    >>>
    >>> hfd, info = nk.fractal_higuchi(signal, k_max='default', show=True)
    >>> hfd #doctest: +SKIP

    Reference
    ----------
    - Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory.
    Physica D: Nonlinear Phenomena, 31(2), 277-283.

    - Vega, C. F., & Noel, J. (2015, June). Parameters analyzed of Higuchi's fractal dimension for EEG brain signals.
    In 2015 Signal Processing Symposium (SPSympo) (pp. 1-5). IEEE. https://ieeexplore.ieee.org/document/7168285
    """

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Get k_max
    if isinstance(k_max, (int, str, list, np.ndarray, pd.Series)):
        k_max, info = complexity_k(signal, k_max=k_max, show=False)

    # Compute slope
    slope, intercept, k_values, average_values = _complexity_k_slope(signal, k_max)

    # Plot
    if show:
        fig = plt.figure(constrained_layout=False)
        fig.suptitle('Higuchi Fractal Dimension (HFD)')
        spec = matplotlib.gridspec.GridSpec(
                ncols=1, nrows=2
            )
        ax_slope = fig.add_subplot(spec[0, 0])
        _fractal_higuchi_plot(k_values, average_values, k_max, slope, intercept, ax=ax_slope)
        ax_kmax = fig.add_subplot(spec[1, 0])
        _complexity_k_plot(info['Values'], info['Scores'], k_max, ax=ax_kmax)

    return slope, {
        "k_max": k_max,
        "k_values": k_values,
        "average_values": average_values,
        "intercept": intercept,
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

    ax.set_title(
        "Least-squares linear best-fit curve for $k_{max}$ = " + str(kmax) + ", slope = " + str(np.round(slope, 2))
    )
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
                label="k = {}".format(i + 1),
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
        ax.plot([], label="k = {}".format(np.min(k_values)), c=colors[0])
        ax.plot([], label="k = {}".format(np.max(k_values)), c=colors[-1])

    fit_values = [slope * i + -intercept for i in -np.log(k_values)]
    ax.plot(-np.log(k_values), fit_values, color="#FF9800", zorder=1)
    ax.legend(loc="lower right")

    return fig
