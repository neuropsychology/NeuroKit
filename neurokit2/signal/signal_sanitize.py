# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def signal_sanitize(signal):
    """**Signal input sanitization**

    Reset indexing for Pandas Series.

    Parameters
    ----------
    signal : Series
        The indexed input signal (``pandas Dataframe.set_index()``)

    Returns
    -------
    Series
        The default indexed signal

    Examples
    --------
    .. ipython:: python

      import pandas as pd
      import neurokit2 as nk

      signal = nk.signal_simulate(duration=10, sampling_rate=1000, frequency=1)
      df = pd.DataFrame({'signal': signal, 'id': [x*2 for x in range(len(signal))]})

      df = df.set_index('id')
      default_index_signal = nk.signal_sanitize(df.signal)


    """

    # Series check for non-default index
    if isinstance(signal, pd.Series) and not isinstance(signal.index, pd.RangeIndex):
        return signal.reset_index(drop=True).values

    return np.array(signal)
