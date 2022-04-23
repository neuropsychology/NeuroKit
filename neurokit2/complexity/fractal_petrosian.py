import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..signal.signal_binarize import _signal_binarize_threshold


def fractal_petrosian(signal, method="C", show=False):
    """**Petrosian fractal dimension (PFD)**

    Petrosian (1995) proposed a fast method to estimate the fractal dimension by converting the
    signal into a binary sequence from which the fractal dimension is estimated. Several variations
    of the algorithm exist (e.g., 'A', 'B', 'C' or 'D'), primarily differing in the way the discrete
    sequence is created.

    **How to transform a continuous signal into a discrete one?**
    * **Method 'A'** binarizes the signal by higher vs. lower values as compated to the signal's
      mean.
    * **Method 'B'** uses values that are within the mean +/- 1 SD band vs. values that are outside
      this band.
    * **Method 'C'** computes the difference between consecutive samples and binarizes depending on
      their sign.
    * **Method 'D'** forms separates consecutive samples that exceed 1 signal's SD from the others
      smaller changes.
    * **Binning**: If an integer *n* is passed, will bin the signal into *n* equal-width bins.

    Most of these methods assume that the signal is periodic (without a linear trend). Linear
    detrending might be useful to eliminate linear trends (see :func:`.signal_detrend`).

    See Also
    --------
    mutual_information, entropy_svd

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    method : str or int
        Method of discretization. Can be one of ``"A"``, ``"B"``, ``"C"``, ``"D"``, an ``int``
        indicating the number of bins, or ``None`` to skip the process (for instance, in cases when
        the binarization has already been done before).
    show : bool
        If ``True``, will show the discrete the signal.

    Returns
    -------
    pfd : float
        The petrosian fractal dimension (PFD).
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute PFD.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=[5, 12])

      @savefig p_fractal_petrosian1.png scale=100%
      pfd, info = nk.fractal_petrosian(signal, method = "A", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      pfd
      info

    .. ipython:: python

      @savefig p_fractal_petrosian2.png scale=100%
      pfd, info = nk.fractal_petrosian(signal, method = "B", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      pfd

    .. ipython:: python

      @savefig p_fractal_petrosian3.png scale=100%
      pfd, info = nk.fractal_petrosian(signal, method = "C", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      pfd

    .. ipython:: python

      signal = nk.signal_simulate(duration=2, frequency=[5], noise = 0.1)

      @savefig p_fractal_petrosian4.png scale=100%
      pfd, info = nk.fractal_petrosian(signal, method = "D", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      pfd

    .. ipython:: python

      @savefig p_fractal_petrosian5.png scale=100%
      pfd, info = nk.fractal_petrosian(signal, method = 6, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      pfd

    References
    ----------
    * Petrosian, A. (1995, June). Kolmogorov complexity of finite sequences and recognition of
      different preictal EEG patterns. In Proceedings eighth IEEE symposium on computer-based
      medical systems (pp. 212-217). IEEE.
    * Kumar, D. K., Arjunan, S. P., & Aliahmad, B. (2017). Fractals: applications in biological
      Signalling and image processing. CRC Press.
    * Goh, C., Hamadicharef, B., Henderson, G., & Ifeachor, E. (2005, June). Comparison of fractal
      dimension algorithms for the computation of EEG biomarkers for dementia. In 2nd International
      Conference on Computational Intelligence in Medicine and Healthcare (CIMED2005).

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Binarize the sequence
    _, n_inversions = _complexity_binarize(signal, method=method, show=show)

    n = len(signal)
    pfd = np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_inversions)))
    return pfd, {"Method": method}


# =============================================================================
# Utils
# =============================================================================
def _complexity_binarize(signal, method="A", show=False):
    """Binarize the signal."""
    if method is None:
        binned = signal
        n_inversions = np.isclose(np.diff(binned), 0).sum()
        if show is True:
            df = pd.DataFrame({"Signal": signal, "Bin": signal, "Index": np.arange(len(signal))})
            df = df.pivot_table(index="Index", columns="Bin", values="Signal")
            for i in df.columns:
                plt.plot(df[i])
    elif isinstance(method, int):
        binned = pd.cut(signal, bins=method, labels=False)
        n_inversions = np.isclose(np.diff(binned), 0).sum()
        if show is True:
            df = pd.DataFrame({"Signal": signal, "Bin": binned, "Index": np.arange(len(signal))})
            df = df.pivot_table(index="Index", columns="Bin", values="Signal")
            for i in df.columns:
                plt.plot(df[i])
            plt.title(f"Method: Binning (bins={method})")
    elif method == "A":
        binned = _signal_binarize_threshold(signal, threshold="mean")
        n_inversions = binned.sum()
        if show is True:
            df = pd.DataFrame({"A": signal, "B": signal})
            df["A"][df["A"] > np.nanmean(signal)] = np.nan
            df["B"][df["B"] <= np.nanmean(signal)] = np.nan
            df.plot()
            plt.axhline(y=np.nanmean(signal), color="r", linestyle="dotted")
            plt.title("Method A")
    elif method == "B":
        m = np.nanmean(signal)
        sd = np.nanstd(signal, ddof=1)
        binned = np.logical_or(signal < m - sd, signal > m + sd)
        n_inversions = binned.sum()
        if show is True:
            df = pd.DataFrame({"A": signal, "B": signal})
            df["A"][np.logical_or(signal < m - sd, signal > m + sd)] = np.nan
            df["B"][~np.isnan(df["A"])] = np.nan
            df.plot()
            plt.axhline(y=m - sd, color="r", linestyle="dotted")
            plt.axhline(y=m + sd, color="r", linestyle="dotted")
            plt.title("Method B")
    elif method == "C":
        binned = np.diff(np.signbit(np.diff(signal)))
        n_inversions = binned.sum()
        if show is True:
            df = pd.DataFrame({"A": signal, "B": signal})
            df["A"][np.insert(np.signbit(np.diff(signal)), 0, False)] = np.nan
            df["B"][~np.isnan(df["A"])] = np.nan
            df.plot()
            plt.title("Method C")

    elif method == "D":
        binned = np.abs(np.diff(signal)) > np.nanstd(signal, ddof=1)
        n_inversions = binned.sum()
        if show is True:
            where = np.where(np.abs(np.diff(signal)) > np.nanstd(signal, ddof=1))[0]
            plt.plot(signal, zorder=1)
            plt.scatter(where, signal[where], color="orange", label="Inversion", zorder=2)
            plt.title("Method D")
    else:
        raise ValueError(
            "`method` must be one of 'A', 'B', 'C' or 'D', or an integer. See the documentation for"
            " more information."
        )

    return binned, n_inversions
