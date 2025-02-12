# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.vq
import scipy.special

from ..misc import check_random_state
from ..stats import standardize
from .optim_complexity_tolerance import complexity_tolerance


def complexity_symbolize(
    signal, method="mean", c=3, random_state=None, show=False, **kwargs
):
    """**Signal Symbolization and Discretization**

    Many complexity indices are made to assess the recurrence and predictability of discrete -
    symbolic - states. As such, continuous signals must be transformed into such discrete sequence.

    For instance, one of the easiest way is to split the signal values into two categories, above
    and below the mean, resulting in a sequence of *A* and *B*. More complex methods have been
    developped to that end.

    * **Method 'A'** binarizes the signal by higher vs. lower values as compated to the signal's
      mean. Equivalent to ``method="mean"`` (``method="median"`` is also valid).
    * **Method 'B'** uses values that are within the mean +/- 1 SD band vs. values that are outside
      this band.
    * **Method 'C'** computes the difference between consecutive samples and binarizes depending on
      their sign.
    * **Method 'D'** forms separates consecutive samples that exceed 1 signal's SD from the others
      smaller changes.
    * **Method 'r'** is based on the concept of :func:`*tolerance* <complexity_tolerance>`, and
      will separate consecutive samples that exceed a given tolerance threshold, by default
      :math:`0.2 * SD`. See :func:`complexity_tolerance` for more details.
    * **Binning**: If an integer *n* is passed, will bin the signal into *n* equal-width bins.
      Requires to specify *c*.
    * **MEP**: Maximum Entropy Partitioning. Requires to specify *c*.
    * **NCDF**: Please help us to improve the documentation here. Requires to specify *c*.
    * **Linear**: Please help us to improve the documentation here. Requires to specify *c*.
    * **Uniform**: Please help us to improve the documentation here. Requires to specify *c*.
    * **kmeans**: k-means clustering. Requires to specify *c*.



    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str or int
        Method of symbolization. Can be one of ``"A"`` (default), ``"B"``, ``"C"``, ``"D"``,
        ``"r"``, ``"Binning"``, ``"MEP"``, ``"NCDF"``, ``"linear"``, ``"uniform"``, ``"kmeans"``,
        ``"equal"``, or ``None`` to skip the process (for instance, in cases when the binarization
        has already been done before).

        See :func:`complexity_symbolize` for details.
    c : int
        Number of symbols *c*, used in some algorithms.
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator. See :func:`misc.check_random_state` for further information.
    show : bool
        Plot the reconstructed attractor. See :func:`complexity_attractor` for details.
    **kwargs
        Other arguments to be passed to :func:`complexity_attractor`.

    Returns
    -------
    array
        A symbolic sequence made of discrete states (e.g., 0 and 1).

    See Also
    ------------
    entropy_shannon, entropy_cumulative_residual, fractal_petrosian

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=[5, 12])

      # Method "A" is equivalent to "mean"
      @savefig p_complexity_symbolize1.png scale=100%
      symbolic = nk.complexity_symbolize(signal, method = "A", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_symbolize2.png scale=100%
      symbolic = nk.complexity_symbolize(signal, method = "B", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_symbolize3.png scale=100%
      symbolic = nk.complexity_symbolize(signal, method = "C", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      signal = nk.signal_simulate(duration=2, frequency=[5], noise = 0.1)

      @savefig p_complexity_symbolize4.png scale=100%
      symbolic = nk.complexity_symbolize(signal, method = "D", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_symbolize5.png scale=100%
      symbolic = nk.complexity_symbolize(signal, method = "r", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_symbolize6.png scale=100%
      symbolic = nk.complexity_symbolize(signal, method = "binning", c=3, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_symbolize7.png scale=100%
      symbolic = nk.complexity_symbolize(signal, method = "MEP", c=3, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_symbolize8.png scale=100%
      symbolic = nk.complexity_symbolize(signal, method = "NCDF", c=3, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_complexity_symbolize9.png scale=100%
      symbolic = nk.complexity_symbolize(signal, method = "linear", c=5, show=True)
      @suppress
      plt.close()


    .. ipython:: python

      @savefig p_complexity_symbolize10.png scale=100%
      symbolic = nk.complexity_symbolize(signal, method = "equal", c=5, show=True)
      @suppress
      plt.close()


    .. ipython:: python

      @savefig p_complexity_symbolize11.png scale=100%
      symbolic = nk.complexity_symbolize(signal, method = "kmeans", c=5, random_state=42, show=True)
      @suppress
      plt.close()

    """
    # Seed the random generator for reproducible results
    rng = check_random_state(random_state)

    # Do nothing
    if method is None:
        symbolic = signal
        if show is True:
            df = pd.DataFrame(
                {"Signal": signal, "Bin": signal, "Index": np.arange(len(signal))}
            )
            df = df.pivot_table(index="Index", columns="Bin", values="Signal")
            for i in df.columns:
                plt.plot(df[i])

    # Binnning
    elif isinstance(method, int):
        c = method
        method = "binning"

    if isinstance(method, str):
        method = method.lower()

        if method in ["a", "mean"]:
            symbolic = (signal > np.nanmean(signal)).astype(int)
            if show is True:
                df = pd.DataFrame({"A": signal, "B": signal})
                df.loc[df["A"] > np.nanmean(signal), "A"] = np.nan
                df.loc[df["B"] <= np.nanmean(signal), "B"] = np.nan
                df.plot()
                plt.axhline(y=np.nanmean(signal), color="r", linestyle="dotted")
                plt.title("Method A")

        elif method == "median":
            symbolic = (signal > np.nanmedian(signal)).astype(int)
            if show is True:
                df = pd.DataFrame({"A": signal, "B": signal})
                df.loc[df["A"] > np.nanmedian(signal), "A"] = np.nan
                df.loc[df["B"] <= np.nanmedian(signal), "B"] = np.nan
                df.plot()
                plt.axhline(y=np.nanmean(signal), color="r", linestyle="dotted")
                plt.title("Binarization by median")

        elif method == "b":
            m = np.nanmean(signal)
            sd = np.nanstd(signal, ddof=1)
            symbolic = np.logical_or(signal < m - sd, signal > m + sd).astype(int)
            if show is True:
                df = pd.DataFrame({"A": signal, "B": signal})
                condition = np.logical_or(signal < m - sd, signal > m + sd)
                df.loc[condition, "A"] = np.nan
                df.loc[~np.isnan(df["A"]), "B"] = np.nan
                df.plot()
                plt.axhline(y=m - sd, color="r", linestyle="dotted")
                plt.axhline(y=m + sd, color="r", linestyle="dotted")
                plt.title("Method B")

        elif method in ["c", "sign"]:
            symbolic = np.signbit(np.diff(signal)).astype(int)
            if show is True:
                df = pd.DataFrame({"A": signal, "B": signal})
                df.loc[np.insert(symbolic, 0, False), "A"] = np.nan
                df.loc[~np.isnan(df["A"]), "B"] = np.nan
                df.plot()
                plt.title("Method C")

        elif method == "d":
            symbolic = (np.abs(np.diff(signal)) > np.nanstd(signal, ddof=1)).astype(int)
            if show is True:
                where = np.where(symbolic)[0]
                plt.plot(signal, zorder=1 == 1)
                plt.scatter(
                    where, signal[where], color="orange", label="Inversion", zorder=2
                )
                plt.title("Method D")

        elif method == "r":
            symbolic = (
                np.abs(np.diff(signal)) > complexity_tolerance(signal, method="sd")[0]
            )
            symbolic = symbolic.astype(int)
            if show is True:
                where = np.where(symbolic == 1)[0]
                plt.plot(signal, zorder=1)
                plt.scatter(
                    where, signal[where], color="orange", label="Inversion", zorder=2
                )
                plt.title("Method based on tolerance r")

        elif method in [
            "binning",
            "mep",
            "ncdf",
            "linear",
            "uniform",
            "kmeans",
            "equal",
        ]:
            n = len(signal)
            if method == "binning":
                symbolic = pd.cut(signal, bins=c, labels=False)

            elif method == "mep":
                Temp = np.hstack(
                    (0, np.ceil(np.arange(1, c) * len(signal) / c) - 1)
                ).astype(int)
                symbolic = np.digitize(signal, np.sort(signal)[Temp])
            elif method == "ncdf":
                symbolic = np.digitize(
                    scipy.special.ndtr(standardize(signal)), np.arange(0, 1, 1 / c)
                )
            elif method == "linear":
                symbolic = np.digitize(
                    signal,
                    np.arange(np.min(signal), np.max(signal), np.ptp(signal) / c),
                )
            elif method == "uniform":
                symbolic = np.zeros(len(signal))
                symbolic[np.argsort(signal)] = np.digitize(
                    np.arange(n), np.arange(0, 2 * n, n / c)
                )
            elif method == "kmeans":
                centroids, labels = scipy.cluster.vq.kmeans2(signal, c, seed=rng)
                labels += 1
                xx = np.argsort(centroids) + 1
                symbolic = np.zeros(n)
                for k in range(1, c + 1):
                    symbolic[labels == xx[k - 1]] = k
            elif method == "equal":
                ix = np.argsort(signal)
                xx = np.round(np.arange(0, 2 * n, n / c)).astype(int)
                symbolic = np.zeros(n)
                for k in range(c):
                    symbolic[ix[xx[k] : xx[k + 1]]] = k + 1

            if show is True:
                df = pd.DataFrame(
                    {"Signal": signal, "Bin": symbolic, "Index": np.arange(len(signal))}
                )
                df = df.pivot_table(index="Index", columns="Bin", values="Signal")
                for i in df.columns:
                    plt.plot(df[i])
                plt.title(f"Method: {method} (c={c})")

        else:
            raise ValueError(
                "`method` must be one of 'A', 'B', 'C' or 'D', 'Binning', 'MEP', 'NCDF', 'linear',"
                " 'uniform', 'kmeans'. See the documentation for more information."
            )

    return symbolic
