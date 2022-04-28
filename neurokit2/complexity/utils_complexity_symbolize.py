# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .optim_complexity_tolerance import complexity_tolerance


def complexity_symbolize(signal, method="mean", show=False, **kwargs):
    """**Signal Symbolization and Discretization**

    Many complexity indices are made to assess the recurrence and predictability of discrete -
    symbolic - states. As such, continuous signals must be transformed into such discrete sequence.

    For instance, one of the easiest way is to split the signal values into two categories, above
    and below the mean, resulting in a sequence of *A* and *B*. More complex methods have been
    developped to that end.

    * **Method 'A'** binarizes the signal by higher vs. lower values as compated to the signal's
      mean.
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

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str or int
        Method of symbolization. Can be one of ``"A"``, ``"B"``, ``"C"``, ``"D"``, ``"r"``, an
        ``int`` indicating the number of bins, or ``None`` to skip the process (for instance, in
        cases when the binarization has already been done before). See :func:`complexity_symbolize`
        for details.
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
      symbolic = nk.complexity_symbolize(signal, method = 10, show=True)
      @suppress
      plt.close()

    """
    # Do nothing
    if method is None:
        symbolic = signal
        if show is True:
            df = pd.DataFrame({"Signal": signal, "Bin": signal, "Index": np.arange(len(signal))})
            df = df.pivot_table(index="Index", columns="Bin", values="Signal")
            for i in df.columns:
                plt.plot(df[i])

    # Binnning
    if isinstance(method, int):
        symbolic = pd.cut(signal, bins=method, labels=False)
        if show is True:
            df = pd.DataFrame({"Signal": signal, "Bin": symbolic, "Index": np.arange(len(signal))})
            df = df.pivot_table(index="Index", columns="Bin", values="Signal")
            for i in df.columns:
                plt.plot(df[i])
            plt.title(f"Method: Binning (bins={method})")
    else:
        method = method.lower()


        if method in ["a", "mean"]:
            symbolic = (signal > np.nanmean(signal)).astype(int)
            if show is True:
                df = pd.DataFrame({"A": signal, "B": signal})
                df["A"][df["A"] > np.nanmean(signal)] = np.nan
                df["B"][df["B"] <= np.nanmean(signal)] = np.nan
                df.plot()
                plt.axhline(y=np.nanmean(signal), color="r", linestyle="dotted")
                plt.title("Method A")

        elif method == "median":
            symbolic = (signal > np.nanmedian(signal)).astype(int)
            if show is True:
                df = pd.DataFrame({"A": signal, "B": signal})
                df["A"][df["A"] > np.nanmedian(signal)] = np.nan
                df["B"][df["B"] <= np.nanmedian(signal)] = np.nan
                df.plot()
                plt.axhline(y=np.nanmean(signal), color="r", linestyle="dotted")
                plt.title("Binarization by median")

        elif method == "b":
            m = np.nanmean(signal)
            sd = np.nanstd(signal, ddof=1)
            symbolic = np.logical_or(signal < m - sd, signal > m + sd).astype(int)
            if show is True:
                df = pd.DataFrame({"A": signal, "B": signal})
                df["A"][np.logical_or(signal < m - sd, signal > m + sd)] = np.nan
                df["B"][~np.isnan(df["A"])] = np.nan
                df.plot()
                plt.axhline(y=m - sd, color="r", linestyle="dotted")
                plt.axhline(y=m + sd, color="r", linestyle="dotted")
                plt.title("Method B")

        elif method == "c":
            symbolic = np.signbit(np.diff(signal))
            if show is True:
                df = pd.DataFrame({"A": signal, "B": signal})
                df["A"][np.insert(symbolic, 0, False)] = np.nan
                df["B"][~np.isnan(df["A"])] = np.nan
                df.plot()
                plt.title("Method C")

        elif method == "d":
            symbolic = np.abs(np.diff(signal)) > np.nanstd(signal, ddof=1)
            if show is True:
                where = np.where(symbolic)[0]
                plt.plot(signal, zorder=1)
                plt.scatter(where, signal[where], color="orange", label="Inversion", zorder=2)
                plt.title("Method D")

        elif method == "r":
            symbolic = np.abs(np.diff(signal)) > complexity_tolerance(signal, method="sd")[0]
            if show is True:
                where = np.where(symbolic)[0]
                plt.plot(signal, zorder=1)
                plt.scatter(where, signal[where], color="orange", label="Inversion", zorder=2)
                plt.title("Method based on tolerance r")

        else:
            raise ValueError(
                "`method` must be one of 'A', 'B', 'C' or 'D', or an integer. See the documentation for"
                " more information."
            )

    return symbolic
