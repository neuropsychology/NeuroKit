# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal import signal_changepoints


def eda_changepoints(eda_cleaned):
    """Calculate the number of change points using of the skin conductance signal in terms of mean and variance.
    Defaults to an algorithm penalty of 10000, as recommended by Halem et al. (2020).

    Parameters
    ----------
    eda_cleaned : Union[list, np.array, pd.Series]
        The cleaned EDA signal.

    Returns
    -------
    float
        Number of changepoints in the

    See Also
    --------
    eda_simulate


    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> # Simulate EDA signal
    >>> eda_signal = nk.eda_simulate(duration=5, scr_number=5, drift=0.1)
    >>> eda_cleaned = nk.eda_clean(eda_signal)
    >>> changepoints = nk.eda_changepoints(eda_cleaned)
    >>> changepoints #doctest: +SKIP

    References
    -----------
    - Halem, S., van Roekel, E., Kroencke, L., Kuper, N., & Denissen, J. (2020). Moments That Matter?
      On the Complexity of Using Triggers Based on Skin Conductance to Sample Arousing Events Within
      an Experience Sampling Framework. European Journal of Personality.

    """
    # Sanity checks
    if not isinstance(eda_cleaned, np.ndarray):
        if isinstance(eda_cleaned, pd.DataFrame):
            colnames = eda_cleaned.columns.values
            if len([i for i in colnames if "EDA_Clean" in i]) == 0:
                raise ValueError(
                    "NeuroKit error: eda_changepoints(): Your input does not contain the cleaned EDA signal."
                )
            else:
                eda_cleaned = eda_cleaned["EDA_Clean"]

        eda_cleaned = np.array(eda_cleaned)

    # Calculate changepoints based on mean and variance
    changepoints = signal_changepoints(eda_cleaned, change="meanvar", show=False, penalty=10000)

    number = len(changepoints)

    return number
