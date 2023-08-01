# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning
from ..stats import mad
from .rsp_fixpeaks import _rsp_fixpeaks_retrieve


def rsp_rav(
    amplitude,
    peaks,
    troughs=None,
    show=False,
):
    """**Respiratory Amplitude Variability (RAV)**

    TODO.

    Parameters
    ----------
    amplitude : Union[list, np.array, pd.Series]
        The amplitude signal as returned by :func:`.rsp_amplitude`.
    peaks : list or array or DataFrame or Series or dict
        The samples at which the inhalation peaks occur. If a dict or a DataFrame is passed, it is
        assumed that these containers were obtained with :func:`.rsp_findpeaks`.
    troughs : list or array or DataFrame or Series or dict
        The samples at which the inhalation troughs occur. If a dict or a DataFrame is passed, it is
        assumed that these containers were obtained with :func:`.rsp_findpeaks`.
    show : bool
        If True, show a plot of the symmetry features.

    Returns
    -------
    pd.DataFrame
        A DataFrame of same length as :func:`.rsp_amplitude` containing the following columns:

        TODO.

    See Also
    --------
    rsp_amplitude, rsp_rrv

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      rsp = nk.rsp_simulate(duration=45, respiratory_rate=15)
      cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
      peak_signal, info = nk.rsp_peaks(cleaned)

      amplitude = nk.rsp_amplitude(cleaned, peaks=peak_signal)


    """
    # Format input.
    peaks, troughs = _rsp_fixpeaks_retrieve(peaks, troughs)
    # nk.signal_plot([cleaned, amplitude], subplots=True)

    # Get values for each cycle
    amplitude_discrete = amplitude[peaks]
    diff_amp = np.diff(amplitude_discrete)

    out = {}  # Initialize empty dict

    # Time domain ------------------------------
    # Mean based
    out["Mean"] = np.nanmean(amplitude_discrete)
    out["SD"] = np.nanstd(amplitude_discrete, ddof=1)

    out["RMSSD"] = np.sqrt(np.mean(diff_amp**2))
    out["SDSD"] = np.nanstd(diff_amp, ddof=1)

    out["CV"] = out["SD"] / out["Mean"]
    out["CVSD"] = out["RMSSD"] / out["Mean"]

    # Robust
    out["Median"] = np.nanmedian(amplitude_discrete)
    out["Mad"] = mad(amplitude_discrete)
    out["MCV"] = out["Mad"] / out["Median"]

    rav = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("RAV_")
