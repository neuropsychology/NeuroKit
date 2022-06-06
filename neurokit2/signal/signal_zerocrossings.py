# -*- coding: utf-8 -*-
import numpy as np


def signal_zerocrossings(signal, direction="both"):
    """**Locate the indices where the signal crosses zero**

    Note that when the signal crosses zero between two points, the first index is returned.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    direction : str
        Direction in which the signal crosses zero, can be ``"positive"``, ``"negative"`` or
        ``"both"`` (default).

    Returns
    -------
    array
        Vector containing the indices of zero crossings.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=5)
      zeros = nk.signal_zerocrossings(signal)
      @savefig p_signal_zerocrossings1.png scale=100%
      nk.events_plot(zeros, signal)
      @suppress
      plt.close()

    .. ipython:: python

      # Only upward or downward zerocrossings
      up = nk.signal_zerocrossings(signal, direction="up")
      down = nk.signal_zerocrossings(signal, direction="down")
      @savefig p_signal_zerocrossings2.png scale=100%
      nk.events_plot([up, down], signal)
      @suppress
      plt.close()

    """
    df = np.diff(np.sign(signal))
    if direction in ["positive", "up"]:
        zerocrossings = np.where(df > 0)[0]
    elif direction in ["negative", "down"]:
        zerocrossings = np.where(df < 0)[0]
    else:
        zerocrossings = np.nonzero(np.abs(df) > 0)[0]

    return zerocrossings
