# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import scipy.spatial
import sklearn.neighbors

from ..stats import density
from .utils_complexity_embedding import complexity_embedding
from .utils_entropy import _entropy_apen


def complexity_tolerance(
    signal, method="maxApEn", r_range=None, delay=None, dimension=None, show=False
):
    """**Automated selection of tolerance (r)**

    Estimate and select the optimal tolerance (*r*) parameter used by other entropy and other
    complexity algorithms.

    Many complexity algorithms are built on the notion of self-similarity and recurrence, and how
    often a system revisits its past states. Considering two states as identical is straightforward
    for discrete systems (e.g., a sequence of ``"A"``, ``"B"`` and ``"C"`` states), but for
    continuous signals, we cannot simply look for when the two numbers are exactly the same.
    Instead, we have to pick a threshold by which to consider two points as similar.

    The tolerance *r* is essentially this threshold value (the numerical difference between two
    similar points that we "tolerate"). This parameter has a critical impact and is a major
    source of inconsistencies in the literature.

    Different methods have been described to estimate the most appropriate tolerance value:

    * **maxApEn**: Different values of tolerance will be tested and the one where the approximate
      entropy (ApEn) is maximized will be selected and returned (Chen, 2008).
    * **recurrence**: The tolerance that yields a recurrence rate (see ``RQA``) close to 1% will
      be returned. Note that this method is currently not suited for very long signals, as it is
      based on a recurrence matrix, which size is close to n^2. Help is needed to address this
      limitation.
    * **neighbours**: The tolerance that yields a number of nearest neighbours (NN) close to 2% will
      be returned.

    As these methods are computationally expensive, other fast heuristics are available:

    * **sd**: r = 0.2 * standard deviation (SD) of the signal will be returned. This is the most
      commonly used value in the literature, though its appropriateness is questionable.
    * **makowski**: Adjusted value based on the SD, the embedding dimension and the signal's
      length. See our `study <https://github.com/DominiqueMakowski/ComplexityTolerance>`_.
    * **nolds**: Adjusted value based on the SD and the dimension. The rationale is that
      the chebyshev distance (used in various metrics) rises logarithmically with increasing
      dimension. ``0.5627 * np.log(dimension) + 1.3334`` is the logarithmic trend line for the
      chebyshev distance of vectors sampled from a univariate normal distribution. A constant of
      ``0.1164`` is used so that ``tolerance = 0.2 * SDs`` for ``dimension = 2`` (originally in
      https://github.com/CSchoel/nolds).
    * **singh2016**: Makes a histogram of the Chebyshev distance matrix and returns the upper bound
      of the modal bin.
    * **chon2009**: Acknowledging that computing multiple ApEns is computationally expensive, Chon
      (2009) suggested an approximation based a heuristic algorithm that takes into account the
      length of the signal, its short-term and long-term variability, and the embedding dimension
      *m*. Initially defined only for *m* in [2-7], we expanded this to work with value of *m*
      (though the accuracy is not guaranteed beyond *m* = 4).


    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str
        Can be ``"maxApEn"`` (default), ``"sd"``, ``"recurrence"``, ``"neighbours"``, ``"nolds"``,
        ``"chon2009"``, or ``"neurokit"``.
    r_range : Union[list, int]
        The range of tolerance values (or the number of values) to test. Only used if ``method`` is
        ``"maxApEn"`` or ``"recurrence"``. If ``None`` (default), the default range will be used;
        ``np.linspace(0.02, 0.8, r_range) * np.std(signal, ddof=1)`` for ``"maxApEn"``, and ``np.
        linspace(0, np.max(d), 30 + 1)[1:]`` for ``"recurrence"``. You can set a lower number for
        faster results.
    delay : int
        Only used if ``method="maxApEn"``. See :func:`entropy_approximate()`.
    dimension : int
        Only used if ``method="maxApEn"``. See :func:`entropy_approximate()`.
    show : bool
        If ``True`` and method is ``"maxApEn"``, will plot the ApEn values for each value of r.

    See Also
    --------
    complexity, complexity_delay, complexity_dimension, complexity_embedding

    Returns
    ----------
    float
        The optimal tolerance value.
    dict
        A dictionary containing additional information.

    Examples
    ----------
    * **Example 1**: The method based on the SD of the signal is fast. The plot shows the d
      distribution of the values making the signal, and the width of the arrow represents the
      chosen ``r`` parameter.

    .. ipython:: python

      import neurokit2 as nk

      # Simulate signal
      signal = nk.signal_simulate(duration=2, frequency=[5, 7, 9, 12, 15])

      # Fast method (based on the standard deviation)
      @savefig p_complexity_tolerance1.png scale=100%
      r, info = nk.complexity_tolerance(signal, method = "sd", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      r

    The dimension can be taken into account:
    .. ipython:: python

      # nolds method
      @savefig p_complexity_tolerance2.png scale=100%
      r, info = nk.complexity_tolerance(signal, method = "nolds", dimension=3, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      r


    * **Example 2**: The method based on the recurrence rate will display the rates according to
      different values of tolerance. The horizontal line indicates 5%.

    .. ipython:: python

      @savefig p_complexity_tolerance3.png scale=100%
      r, info = nk.complexity_tolerance(signal, delay=1, dimension=10,
                                        method = 'recurrence', show=True)
      @suppress
      plt.close()

    .. ipython:: python

      r

    An alternative, better suited for long signals is to use nearest neighbours.

    .. ipython:: python

      @savefig p_complexity_tolerance4.png scale=100%
      r, info = nk.complexity_tolerance(signal, delay=1, dimension=10,
                                        method = 'neighbours', show=True)
      @suppress
      plt.close()

    Another option is to use the density of distances.

    .. ipython:: python

      @savefig p_complexity_tolerance5.png scale=100%
      r, info = nk.complexity_tolerance(signal, delay=1, dimension=3,
                                        method = 'bin', show=True)
      @suppress
      plt.close()

    * **Example 3**: The default method selects the tolerance at which *ApEn* is maximized.

    .. ipython:: python

      # Slow method
      @savefig p_complexity_tolerance6.png scale=100%
      r, info = nk.complexity_tolerance(signal, delay=8, dimension=6,
                                        method = 'maxApEn', show=True)
      @suppress
      plt.close()

    .. ipython:: python

      r

    * **Example 4**: The tolerance values that are tested can be modified to get a more precise
      estimate.

    .. ipython:: python

      # Narrower range
      @savefig p_complexity_tolerance7.png scale=100%
      r, info = nk.complexity_tolerance(signal, delay=8, dimension=6, method = 'maxApEn',
                                        r_range=np.linspace(0.002, 0.8, 30), show=True)
      @suppress
      plt.close()

    .. ipython:: python

      r

    References
    -----------
    * Chon, K. H., Scully, C. G., & Lu, S. (2009). Approximate entropy for all signals. IEEE
      engineering in medicine and biology magazine, 28(6), 18-23.
    * Lu, S., Chen, X., Kanters, J. K., Solomon, I. C., & Chon, K. H. (2008). Automatic selection of
      the threshold value r for approximate entropy. IEEE Transactions on Biomedical Engineering,
      55(8), 1966-1972.
    * Chen, X., Solomon, I. C., & Chon, K. H. (2008). Parameter selection criteria in approximate
      entropy and sample entropy with application to neural respiratory signals. Am. J. Physiol.
      Regul. Integr. Comp. Physiol.
    * Singh, A., Saini, B. S., & Singh, D. (2016). An alternative approach to approximate entropy
      threshold value (r) selection: application to heart rate variability and systolic blood
      pressure variability under postural challenge. Medical & biological engineering & computing,
      54(5), 723-732.

    """
    if not isinstance(method, str):
        return method, {"Method": "None"}

    # Method
    method = method.lower()
    if method in ["traditional", "sd", "std", "default"]:
        r = 0.2 * np.std(signal, ddof=1)
        info = {"Method": "20% SD"}

    elif method in ["adjusted_sd", "nolds"] and (
        isinstance(dimension, (int, float)) or dimension is None
    ):
        if dimension is None:
            raise ValueError("'dimension' cannot be empty for the 'nolds' method.")
        r = (
            0.11604738531196232
            * np.std(signal, ddof=1)
            * (0.5627 * np.log(dimension) + 1.3334)
        )
        info = {"Method": "Adjusted 20% SD"}

    elif method in ["chon", "chon2009"] and (
        isinstance(dimension, (int, float)) or dimension is None
    ):
        if dimension is None:
            raise ValueError("'dimension' cannot be empty for the 'chon2009' method.")
        sd1 = np.std(np.diff(signal), ddof=1)  # short-term variability
        sd2 = np.std(signal, ddof=1)  # long-term variability of the signal

        # Here are the 3 formulas from Chon (2009):
        # For m=2: r =(−0.036 + 0.26 * sqrt(sd1/sd2)) / (len(signal) / 1000)**1/4
        # For m=3: r =(−0.08 + 0.46 * sqrt(sd1/sd2)) / (len(signal) / 1000)**1/4
        # For m=4: r =(−0.12 + 0.62 * sqrt(sd1/sd2)) / (len(signal) / 1000)**1/4
        # For m=5: r =(−0.16 + 0.78 * sqrt(sd1/sd2)) / (len(signal) / 1000)**1/4
        # For m=6: r =(−0.19 + 0.91 * sqrt(sd1/sd2)) / (len(signal) / 1000)**1/4
        # For m=7: r =(−0.2 + 1 * sqrt(sd1/sd2)) / (len(signal) / 1000)**1/4
        if dimension <= 2 and dimension <= 7:
            x = [-0.036, -0.08, -0.12, -0.16, -0.19, -0.2][dimension - 2]
            y = [0.26, 0.46, 0.62, 0.78, 0.91, 1][dimension - 2]
        else:
            # We need to extrapolate the 2 first numbers, x and y
            # np.polyfit(np.log([2,3,4, 5, 6, 7]), [-0.036, -0.08, -0.12, -0.16, -0.19, -0.2], 1)
            # np.polyfit([2,3,4, 5, 6, 7], [0.26, 0.46, 0.62, 0.78, 0.91, 1], 1)
            x = -0.034 * dimension + 0.022
            y = 0.14885714 * dimension - 0.00180952

        r = (x + y * np.sqrt(sd1 / sd2)) / (len(signal) / 1000) ** 1 / 4
        info = {"Method": "Chon (2009)"}

    elif method in ["neurokit", "makowski"] and (
        isinstance(dimension, (int, float)) or dimension is None
    ):
        # Method described in
        # https://github.com/DominiqueMakowski/ComplexityTolerance
        if dimension is None:
            raise ValueError("'dimension' cannot be empty for the 'makowski' method.")
        n = len(signal)
        r = np.std(signal, ddof=1) * (
            0.2811 * (dimension - 1)
            + 0.0049 * np.log(n)
            - 0.02 * ((dimension - 1) * np.log(n))
        )

        info = {"Method": "Makowski"}

    elif method in ["maxapen", "optimize"]:
        r, info = _optimize_tolerance_maxapen(
            signal, r_range=r_range, delay=delay, dimension=dimension
        )
        info.update({"Method": "Max ApEn"})

    elif method in ["recurrence", "rqa"]:
        r, info = _optimize_tolerance_recurrence(
            signal, r_range=r_range, delay=delay, dimension=dimension
        )
        info.update({"Method": "1% Recurrence Rate"})

    elif method in ["neighbours", "neighbors", "nn"]:
        r, info = _optimize_tolerance_neighbours(
            signal, r_range=r_range, delay=delay, dimension=dimension
        )
        info.update({"Method": "2% Neighbours"})

    elif method in ["bin", "bins", "singh", "singh2016"]:
        r, info = _optimize_tolerance_bin(signal, delay=delay, dimension=dimension)
        info.update({"Method": "bin"})

    else:
        raise ValueError(
            "NeuroKit error: complexity_tolerance(): 'method' not recognized."
        )

    if show is True:
        _optimize_tolerance_plot(r, info, method=method, signal=signal)
    return r, info


# =============================================================================
# Internals
# =============================================================================


def _optimize_tolerance_recurrence(signal, r_range=None, delay=None, dimension=None):
    # Optimize missing parameters
    if delay is None or dimension is None:
        raise ValueError(
            "If method='recurrence', both delay and dimension must be specified."
        )

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
    # Closest to 0.01 (1%)
    optimal = r_range[np.abs(recurrence_rate - 0.01).argmin()]

    return optimal, {"Values": r_range, "Scores": recurrence_rate}


def _optimize_tolerance_maxapen(signal, r_range=None, delay=None, dimension=None):
    # Optimize missing parameters
    if delay is None or dimension is None:
        raise ValueError(
            "If method='maxApEn', both delay and dimension must be specified."
        )

    if r_range is None:
        r_range = 40
    if isinstance(r_range, int):
        r_range = np.linspace(0.02, 0.8, r_range) * np.std(signal, ddof=1)

    apens = np.zeros_like(r_range)
    info = {"kdtree1": None, "kdtree2": None}
    for i, r in enumerate(r_range):
        apens[i], info = _entropy_apen(
            signal,
            delay=delay,
            dimension=dimension,
            tolerance=r,
            kdtree1=info["kdtree1"],
            kdtree2=info["kdtree2"],
        )
    # apens = [_entropy_apen(signal, delay=delay, dimension=dimension, tolerance=r) for r in r_range]

    return r_range[np.argmax(apens)], {"Values": r_range, "Scores": np.array(apens)}


def _optimize_tolerance_neighbours(signal, r_range=None, delay=None, dimension=None):
    if delay is None:
        delay = 1
    if dimension is None:
        dimension = 1
    if r_range is None:
        r_range = 50
    if isinstance(r_range, int):
        r_range = np.linspace(0.02, 0.8, r_range) * np.std(signal, ddof=1)

    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)

    kdtree = sklearn.neighbors.KDTree(embedded, metric="chebyshev")
    counts = np.array(
        [
            np.mean(
                kdtree.query_radius(embedded, r, count_only=True).astype(np.float64)
                / embedded.shape[0]
            )
            for r in r_range
        ]
    )
    # Closest to 0.02 (2%)
    optimal = r_range[np.abs(counts - 0.02).argmin()]
    return optimal, {"Values": r_range, "Scores": counts}


def _optimize_tolerance_bin(signal, delay=None, dimension=None):
    # Optimize missing parameters
    if delay is None or dimension is None:
        raise ValueError("If method='bin', both delay and dimension must be specified.")

    # Compute distance matrix
    emb = complexity_embedding(signal, delay=delay, dimension=dimension)
    d = scipy.spatial.distance.cdist(emb, emb, metric="chebyshev")

    # Histogram of the lower triangular (without the diagonal)
    y, x = np.histogram(d[np.tril_indices(len(d), k=-1)], bins=200, density=True)

    # Most common distance
    # Divide by two because r corresponds to the radius of the circle (NOTE: this is
    # NOT in the paper and thus, opinion is required!)
    optimal = x[np.argmax(y) + 1] / 2

    return optimal, {"Values": x[1::] / 2, "Scores": y}


# =============================================================================
# Plotting
# =============================================================================
def _optimize_tolerance_plot(r, info, ax=None, method="maxApEn", signal=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if method in [
        "traditional",
        "sd",
        "std",
        "default",
        "none",
        "adjusted_sd",
        "nolds",
        "chon",
        "chon2009",
    ]:
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

    if method in ["bin", "bins", "singh", "singh2016"]:
        ax.set_title("Optimization of Tolerance Threshold (r)")
        ax.set_xlabel("Chebyshev Distance")
        ax.set_ylabel("Density")
        ax.plot(info["Values"], info["Scores"], color="#4CAF50")
        ax.axvline(x=r, color="#E91E63", label="Optimal r: " + str(np.round(r, 3)))
        ax.legend(loc="upper right")

        return fig

    r_range = info["Values"]
    y_values = info["Scores"]

    # Custom legend depending on method
    if method in ["maxapen", "optimize"]:
        ylabel = "Approximate Entropy $ApEn$"
        legend = "$ApEn$"
    else:
        y_values *= 100  # Convert to percentage
        ax.axhline(y=0.5, color="grey")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        if method in ["neighbours", "neighbors", "nn"]:
            ylabel = "Nearest Neighbours"
            legend = "$NN$"
        else:
            ylabel = "Recurrence Rate $RR$"
            legend = "$RR$"

    ax.set_title("Optimization of Tolerance Threshold (r)")
    ax.set_xlabel("Tolerance threshold $r$")
    ax.set_ylabel(ylabel)
    ax.plot(r_range, y_values, "o-", label=legend, color="#80059c")
    ax.axvline(x=r, color="#E91E63", label="Optimal r: " + str(np.round(r, 3)))
    ax.legend(loc="upper right")

    return fig
