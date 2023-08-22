# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .rsp_fixpeaks import _rsp_fixpeaks_retrieve


def rsp_rav(
    amplitude,
    peaks,
    troughs=None,
):
    """**Respiratory Amplitude Variability (RAV)**

    Computes indices of amplitude variability, such as the mean and SD of the amplitude, and the
    RMSSD of the successive differences.

    .. note::

      This is an exploratory feature. If you manage to find studies and literature on RAV, please
      let us know by opening an issue on GitHub. Adding more indices (similar to HRV) would be
      trivial, but having some evidence as for its usefulness would be prerequisite.

    Parameters
    ----------
    amplitude : Union[list, np.array, pd.Series]
        The amplitude signal as returned by :func:`.rsp_amplitude`.
    peaks : list or array or DataFrame or Series or dict
        The samples at which the inhalation peaks occur. If a dict or a DataFrame is passed, it is
        assumed that these containers were obtained with :func:`.rsp_findpeaks`.
    troughs : list or array or DataFrame or Series or dict
        The samples at which the inhalation troughs occur. If a dict or a DataFrame is passed, it is
        assumed that these containers were obtained with :func:`.rsp_findpeaks`. This argument can
        be inferred from the ``peaks`` argument if the information.

    Returns
    -------
    pd.DataFrame
        A DataFrame of containing the following columns with RAV indices.

    See Also
    --------
    rsp_amplitude, rsp_rrv

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      rsp = nk.rsp_simulate(duration=45, respiratory_rate=15)
      cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
      peak_signal, info = nk.rsp_peaks(cleaned, sampling_rate=1000)

      amplitude = nk.rsp_amplitude(cleaned, peaks=peak_signal)

      rav = nk.rsp_rav(amplitude, peaks=peak_signal)
      rav


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
    # out["SDSD"] = np.nanstd(diff_amp, ddof=1)

    # out["CV"] = out["SD"] / out["Mean"]
    out["CVSD"] = out["RMSSD"] / out["Mean"]

    # # Robust
    # out["Median"] = np.nanmedian(amplitude_discrete)
    # out["Mad"] = mad(amplitude_discrete)
    # out["MCV"] = out["Mad"] / out["Median"]

    return pd.DataFrame.from_dict(out, orient="index").T.add_prefix("RAV_")
