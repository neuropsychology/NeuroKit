import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import scipy.spatial

from .optim_complexity_tolerance import complexity_tolerance
from .utils_complexity_embedding import complexity_embedding


def recurrence_matrix(signal, delay=1, dimension=3, tolerance="default", show=False):
    """**Recurrence Matrix**

    Fast Python implementation of recurrence matrix (tested against pyRQA). Returns a tuple
    with the recurrence matrix (made of 0s and 1s) and the distance matrix (the non-binarized
    version of the former).

    It is used in :func:`Recurrence Quantification Analysis (RQA) <complexity_rqa>`.

    Parameters
    ----------
    signal : Union[list, np.ndarray, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    tolerance : float
        Tolerance (often denoted as *r*), distance to consider two data points as similar. If
        ``"sd"`` (default), will be set to :math:`0.2 * SD_{signal}`. See
        :func:`complexity_tolerance` to estimate the optimal value for this parameter. A rule of
        thumb is to set *r* so that the percentage of points classified as recurrences is about
        2-5%.
    show : bool
        Visualise recurrence matrix.

    See Also
    --------
    complexity_embedding, complexity_delay, complexity_dimension, complexity_tolerance,
    complexity_rqa

    Returns
    -------
    np.ndarray
        The recurrence matrix.
    np.ndarray
        The distance matrix.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, sampling_rate=100, frequency=[5, 6], noise=0.01)

      # Default r
      @savefig p_recurrence_matrix1.png scale=100%
      rc, _ = nk.recurrence_matrix(signal, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      # Larger radius
      @savefig p_recurrence_matrix2.png scale=100%
      rc, d = nk.recurrence_matrix(signal, tolerance=0.5, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      # Optimization of tolerance via recurrence matrix
      @savefig p_recurrence_matrix3.png scale=100%
      tol, _ = nk.complexity_tolerance(signal, dimension=1, delay=3, method="recurrence", show=True)
      @suppress
      plt.close()

    .. ipython:: python

      @savefig p_recurrence_matrix4.png scale=100%
      rc, d = nk.recurrence_matrix(signal, tolerance=tol, show=True)
      @suppress
      plt.close()


    References
    ----------
    * Rawald, T., Sips, M., Marwan, N., & Dransch, D. (2014). Fast computation of recurrences
      in long time series. In Translational Recurrences (pp. 17-29). Springer, Cham.
    * Dabiré, H., Mestivier, D., Jarnet, J., Safar, M. E., & Chau, N. P. (1998). Quantification of
      sympathetic and parasympathetic tones by nonlinear indexes in normotensive rats. American
      Journal of Physiology-Heart and Circulatory Physiology, 275(4), H1290-H1297.
    """
    tolerance, _ = complexity_tolerance(
        signal, method=tolerance, delay=delay, dimension=dimension, show=False
    )

    # Time-delay embedding
    emb = complexity_embedding(signal, delay=delay, dimension=dimension)

    # Compute distance matrix
    d = scipy.spatial.distance.cdist(emb, emb, metric="euclidean")

    # Initialize the recurrence matrix filled with 0s
    recmat = np.zeros((len(d), len(d)))
    # If lower than tolerance, then 1
    recmat[d <= tolerance] = 1

    # Plotting
    if show is True:
        try:
            fig, axes = plt.subplots(ncols=2)
            axes[0].imshow(recmat, cmap="Greys")
            axes[0].set_title("Recurrence Matrix")
            im = axes[1].imshow(d)
            axes[1].set_title("Distance")
            cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.ax.plot([0, 1], [tolerance] * 2, color="r")
            # Flip the matrix to match traditional RQA representation
            axes[0].invert_yaxis()
            axes[1].invert_yaxis()
            axes[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            axes[1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        except MemoryError as e:
            raise MemoryError(
                "NeuroKit error: complexity_rqa(): the recurrence plot is too large to display. ",
                "You can recover the matrix from the parameters and try to display parts of it.",
            ) from e

    return recmat, d
