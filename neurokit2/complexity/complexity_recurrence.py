import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

from .complexity_embedding import complexity_embedding
from .optim_complexity_tolerance import complexity_tolerance


def complexity_recurrence(signal, delay=1, dimension=3, tolerance="default", show=False):
    """Recurrence matrix (Python implementation)

    Fast Python implementation of recurrence matrix (tested against pyRQA). Returns a tuple
    with the recurrence matrix (made of 0s and 1s) and the distance matrix (the non-binarized
    version of the former).

    Parameters
    ----------
    signal : Union[list, np.ndarray, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common
        to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003),
        or to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding
        returns an array with two columns corresponding to the original signal and its delayed (by
        Tau) version.
    tolerance : float
        Tolerance (similarity threshold, often denoted as 'r'). The radius used for detecting
        neighbours (states considered as recurrent). A rule of thumb is to set 'r' so that the
        percentage of points classified as recurrences is about 2-5%.
    show : bool
        Visualise recurrence matrix.

    See Also
    --------
    complexity_embedding, complexity_tolerance

    Returns
    -------
    np.ndarray
        The recurrence matrix.
    np.ndarray
        The distance matrix.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=5, sampling_rate=100, frequency=[5, 6], noise=0.01)
    >>>
    >>> # Default r
    >>> rc, _ = nk.complexity_recurrence(signal, show=True)
    >>>
    >>> # Larger radius
    >>> rc, d = nk.complexity_recurrence(signal, tolerance=0.5, show=True)
    >>>
    >>> # Optimization of tolerance via recurrence matrix
    >>> rc, d = nk.complexity_tolerance(signal, delay=1, dimension=3, method="recurrence", show=True)



    References
    ----------
    - Rawald, T., Sips, M., Marwan, N., & Dransch, D. (2014). Fast computation of recurrences
    in long time series. In Translational Recurrences (pp. 17-29). Springer, Cham.
    - Dabiré, H., Mestivier, D., Jarnet, J., Safar, M. E., & Chau, N. P. (1998). Quantification of
    sympathetic and parasympathetic tones by nonlinear indexes in normotensive rats. American
    Journal of Physiology-Heart and Circulatory Physiology, 275(4), H1290-H1297.
    """
    if tolerance == "default":
        tolerance, _ = complexity_tolerance(
            signal, method="sd", delay=None, dimension=None, show=False
        )

    # Time-delay embedding
    emb = complexity_embedding(signal, delay=delay, dimension=dimension)

    # Compute distance matrix
    d = scipy.spatial.distance.cdist(emb, emb, metric="euclidean")

    # Flip the matrix to match traditional RQA representation
    d = np.flip(d, axis=0)

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
        except MemoryError as e:
            raise MemoryError(
                "NeuroKit error: complexity_rqa(): the recurrence plot is too large to display. ",
                "You can recover the matrix from the parameters and try to display parts of it.",
            ) from e

    return recmat, d
