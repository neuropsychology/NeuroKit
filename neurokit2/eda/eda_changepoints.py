# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal import signal_changepoints


def eda_changepoints(eda_cleaned, penalty=10000, show=False):
    """**Calculate Number of Change Points**

    Calculate the number of change points using of the skin conductance signal in terms of mean
    and variance.

    .. note::

        This function is somewhat experimental, and improvements or discussions about it are
        welcome.

    Parameters
    ----------
    eda_cleaned : Union[list, np.array, pd.Series]
        The cleaned EDA signal.
    penalty : int
        Defaults to an algorithm penalty of 10000, as recommended by Halem et al. (2020). See
        :func:`.signal_changepoints`.
    show : bool
        Show the signal with the change points.

    Returns
    -------
    float
        Number of changepoints in the

    See Also
    --------
    eda_simulate, .signal_changepoints


    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate EDA signal
      eda_signal = nk.eda_simulate(duration=5, sampling_rate=100, scr_number=5, drift=0.1)
      eda_cleaned = nk.eda_clean(eda_signal, sampling_rate=100)

      @savefig p_eda_changepoints1.png scale=100%
      nk.eda_changepoints(eda_cleaned, penalty = 100, show=True)
      @suppress
      plt.close()

    References
    -----------
    * van Halem, S., Van Roekel, E., Kroencke, L., Kuper, N., & Denissen, J. (2020). Moments that
      matter? On the complexity of using triggers based on skin conductance to sample arousing
      events within an experience sampling framework. European Journal of Personality, 34(5),
      794-807.

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
    changepoints = signal_changepoints(eda_cleaned, change="meanvar", show=show, penalty=penalty)

    number = len(changepoints)

    return number
