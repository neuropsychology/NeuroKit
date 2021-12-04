import numpy as np

from .complexity_embedding import complexity_embedding
from .optim_complexity_tolerance import complexity_tolerance


def complexity_recurrence(signal, delay=1, dimension=10, tolerance="default", show=False):
    """Recurrence matrix (Python implementation)

    Fast pure Python implementation of recurrence matrix (tested against pyrqa).

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

    Returns
    -------
    np.ndarray
        The recurrence matrix.

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
    emb = complexity_embedding(signal, delay=1, dimension=10)

    # Initialize the 3D matrices
    x = np.zeros((len(emb), len(emb), dimension))
    y = np.zeros((len(emb), len(emb), dimension))

    # Iterate over the lower triangle only
    # TODO: this could probably be done faster via some form of vectorization
    for i in range(len(emb)):
        for ii in range(i + 1):
            x[i, ii, :] = emb[i, :]
            y[i, ii, :] = emb[ii, :]

    # Compute distance between x and y
    d = np.sqrt(np.sum(np.square(np.diff([y, x], axis=0)[0]), axis=-1))
    # Initialize the recurrence matrix filled with 0s
    rc = np.zeros((len(emb), len(emb)))
    # If lower than tolerance, then 1
    rc[d <= tolerance] = 1
    # Copy lower triangle to upper
    upper_triangle = np.triu_indices(len(rc), 0)
    rc[upper_triangle] = rc.T[upper_triangle]
    # Flip the matrix
    return np.flip(rc, axis=0)
