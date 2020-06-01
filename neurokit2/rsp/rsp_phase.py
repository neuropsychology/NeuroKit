# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal import signal_phase
from .rsp_fixpeaks import _rsp_fixpeaks_retrieve


def rsp_phase(peaks, troughs=None, desired_length=None):
    """
    Compute respiratory phase (inspiration and expiration).

    Finds the respiratory phase, labelled as 1 for inspiration and 0 for expiration.
    Parameters
    ----------
    peaks, troughs : list, array, DataFrame, Series or dict
        The samples at which the inhalation peaks occur. If a dict or a
        DataFrame is passed, it is assumed that these containers were obtained
        with `rsp_findpeaks()`.
    desired_length : int
        By default, the returned respiration rate has the same number of
        elements as `peaks`. If set to an integer, the returned rate will be
        interpolated between `peaks` over `desired_length` samples. Has no
        effect if a DataFrame is passed in as the `peaks` argument.

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as `rsp_signal` containing the following
        columns:

        - *"RSP_Inspiration"*: breathing phase, marked by "1" for inspiration
          and "0" for expiration.
        - *"RSP_Phase_Completion"*: breathing phase completion, expressed in
          percentage (from 0 to 1), representing the stage of the current
          respiratory phase.

    See Also
    --------
    rsp_clean, rsp_peaks, rsp_amplitude, rsp_process, rsp_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> rsp = nk.rsp_simulate(duration=30, respiratory_rate=15)
    >>> cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
    >>> peak_signal, info = nk.rsp_peaks(cleaned)
    >>>
    >>> phase = nk.rsp_phase(peak_signal)
    >>> fig = nk.signal_plot([rsp, phase], standardize=True)
    >>> fig #doctest: +SKIP

    """
    # Format input.
    peaks, troughs, desired_length = _rsp_fixpeaks_retrieve(peaks, troughs, desired_length)

    # Phase
    inspiration = np.full(desired_length, np.nan)
    inspiration[peaks] = 0.0
    inspiration[troughs] = 1.0

    last_element = np.where(~np.isnan(inspiration))[0][-1]  # Avoid filling beyond the last peak/trough
    inspiration[0:last_element] = pd.Series(inspiration).fillna(method="pad").values[0:last_element]

    # Phase Completion
    completion = signal_phase(inspiration, method="percent")

    out = pd.DataFrame({"RSP_Phase": inspiration, "RSP_Phase_Completion": completion})

    return out
