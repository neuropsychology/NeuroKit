import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from .entropy_shannon import entropy_shannon
from .optim_complexity_tolerance import complexity_tolerance
from .utils_complexity_embedding import complexity_embedding


def fractal_density(signal, delay=1, tolerance="sd", bins=None, show=False, **kwargs):
    """**Density Fractal Dimension (DFD)**

    This is a **Work in Progress (WIP)**. The idea is to find a way of, essentially, averaging
    attractors. Because one can not directly average the trajectories, one way is to convert the
    attractor to a 2D density matrix that we can use similarly to a time-frequency heatmap. However,
    it is very unclear how to then derive meaningful indices from this density plot. Also, how many
    bins, or smoothing, should one use?

    Basically, this index is exploratory and should not be used in its state. However, if you're
    interested in the problem of "average" attractors (e.g., from multiple epochs / trials), and
    you want to think about it with us, feel free to let us know!

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    tolerance : float
        Tolerance (often denoted as *r*), distance to consider two data points as similar. If
        ``"sd"`` (default), will be set to :math:`0.2 * SD_{signal}`. See
        :func:`complexity_tolerance` to estimate the optimal value for this parameter.
    bins : int
        If not ``None`` but an integer, will use this value for the number of bins instead of a
        value based on the ``tolerance`` parameter.
    show : bool
        Plot the density matrix. Defaults to ``False``.
    **kwargs
        Other arguments to be passe.

    Returns
    ---------
    dfd : float
        The density fractal dimension.
    info : dict
        A dictionary containing additional information.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=[5, 9], noise=0.01)

      @savefig p_fractal_density1.png scale=100%
      dfd, _ = nk.fractal_density(signal, delay=20, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      signal = nk.signal_simulate(duration=4, frequency=[5, 10, 11], noise=0.01)
      epochs = nk.epochs_create(signal, events=20)
      @savefig p_fractal_density2.png scale=100%
      dfd, info1 = nk.fractal_density(epochs, delay=20, bins=20, show=True)
      @suppress
      plt.close()


    Compare the complexity of two signals.

    .. warning::

        Help is needed to find a way to make statistics and comparing two density maps.


    .. ipython:: python

      import matplotlib.pyplot as plt

      sig2 = nk.signal_simulate(duration=4, frequency=[4, 12, 14], noise=0.01)
      epochs2 = nk.epochs_create(sig2, events=20)
      dfd, info2 = nk.fractal_density(epochs2, delay=20, bins=20)

      # Difference between two density maps
      D = info1["Average"] - info2["Average"]

      @savefig p_fractal_density3.png scale=100%
      plt.imshow(nk.standardize(D), cmap='RdBu')
      @suppress
      plt.close()

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )
    if isinstance(signal, (np.ndarray, list)):
        # This index is made to work on epochs, so if not an epoch,
        # got to transform
        signal = {"1": pd.DataFrame({"Signal": signal})}

    # Get edges and tolerance from first epoch. Imperfect but what to do?
    edges = np.percentile(signal["1"]["Signal"].values, [1, 99])

    if bins is None:
        tolerance, _ = complexity_tolerance(signal["1"]["Signal"].values, method="sd")

        # Compute number of "bins"
        bins = int((edges[1] - edges[0]) / tolerance)


    # Prepare the container for the 2D density matrix
    X = np.empty((bins, bins, len(signal)))
    for i, (k, epoch) in enumerate(signal.items()):
        X[:, :, i] = _fractal_density(
            epoch["Signal"].dropna().values, edges, bins=bins, delay=delay, **kwargs
        )

    # Compute grand average
    grand_av = np.mean(X, axis=-1)

    # Compute DFD
    freq, x = np.histogram(grand_av, bins=bins)
    dfd, _ = entropy_shannon(freq=freq)

    if show is True:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(grand_av)
        ax[1].bar(x[1::] - np.diff(x) / 2, height=freq, width=np.diff(x))

    return dfd, {"Density": X, "Average": grand_av}


# =============================================================================
# Utilities
# =============================================================================
def _fractal_density(signal, edges, bins, delay=1, method="histogram"):
    emb = complexity_embedding(signal, delay=delay, dimension=2)

    if method == "histogram":
        edges = np.linspace(edges[0], edges[1], bins + 1)
        edges = np.reshape(np.repeat(edges, 2), (len(edges), 2))
        X, _, = np.histogramdd(
            emb,
            bins=edges.T,
            density=False,
        )
    else:
        kde = scipy.stats.gaussian_kde(emb.T)
        kde.set_bandwidth(bw_method=(edges[1] - edges[0]) / bins)
        # Create grid
        edges = np.linspace(edges[0], edges[1], bins)
        x, y = np.meshgrid(edges, edges)
        grid = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)

        X = np.reshape(kde(grid.T), (len(edges), len(edges)))

    return np.log(1 + X)
