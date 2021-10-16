import numpy as np
import pandas as pd


def complexity_rqa(signal):
    """Recurrence quantification analysis (RQA)

    Recurrence quantification analysis (RQA) is a method of complexity analysis
    for the investigation of dynamical systems. It quantifies the number and duration
    of recurrences of a dynamical system presented by its phase space trajectory.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> x = [1, 2, 3, 4, 5]
    >>> autocor = nk.signal_autocor(x)
    >>> autocor #doctest: +SKIP
    """
    # Try loading mne
    try:
        import pyrqa
    except ImportError as e:
        raise ImportError(
            "NeuroKit error: complexity_rqa(): the 'pyrqa' module is required for this function to run. ",
            "Please install it first (`pip install PyRQA`).",
        ) from e
