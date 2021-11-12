import numpy as np
import pandas as pd

from .complexity_embedding import complexity_embedding


def fisher_information(signal, delay=1, dimension=2):
    """Fisher Information (FI)

    The Fisher information was introduced by R. A. Fisher in 1925, as a measure of "intrinsic accuracy"
    in statistical estimation theory. As the Shannon entropy, it can be employed as a quality of an
    efficient measurement procedure, used to estimate a system's disorder.

    See Also
    --------
    mutual_information, entropy_svd

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common
        to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or
        to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns
        an array with two columns corresponding to the original signal and its delayed (by Tau) version.

    Returns
    -------
    fi : float
        The computed fisher information measure.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute fisher information.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=2, frequency=5)
    >>>
    >>> fi, info = nk.fisher_information(signal, delay=10, dimension=3)

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
    W = np.linalg.svd(embedded, compute_uv=False)
    W /= np.sum(W)  # normalize singular values
    FI_v = (W[1:] - W[:-1]) ** 2 / W[:-1]

    return np.sum(FI_v), {"Dimension": dimension, "Delay": delay, "Values": FI_v}
