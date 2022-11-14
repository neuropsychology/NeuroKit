# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning, find_closest
from ..signal import signal_interpolate
from ..stats import rescale
from .rsp_fixpeaks import _rsp_fixpeaks_retrieve


def rsp_symmetry(
    rsp_cleaned,
    peaks,
    troughs=None,
    interpolation_method="monotone_cubic",
    show=False,
):
    """**Respiration Cycle Symmetry Features**

    Compute symmetry features of the respiration cycle, such as the Peak-Trough symmetry and the
    Rise-Decay symmetry (see Cole, 2019). Note that the values for each cycle are interpolated to
    the same length as the signal (and the first and last cycles, for which one cannot compute the
    symmetry characteristics, are padded).

    .. figure:: ../img/cole2019.png
       :alt: Figure from Cole and Voytek (2019).
       :target: https://journals.physiology.org/doi/full/10.1152/jn.00273.2019

    Parameters
    ----------
    rsp_cleaned : Union[list, np.array, pd.Series]
        The cleaned respiration channel as returned by :func:`.rsp_clean`.
    peaks : list or array or DataFrame or Series or dict
        The samples at which the inhalation peaks occur. If a dict or a DataFrame is passed, it is
        assumed that these containers were obtained with :func:`.rsp_findpeaks`.
    troughs : list or array or DataFrame or Series or dict
        The samples at which the inhalation troughs occur. If a dict or a DataFrame is passed, it is
        assumed that these containers were obtained with :func:`.rsp_findpeaks`.
    interpolation_method : str
        Method used to interpolate the amplitude between peaks. See :func:`.signal_interpolate`.
        ``"monotone_cubic"`` is chosen as the default interpolation method since it ensures monotone
        interpolation between data point (i.e., it prevents physiologically implausible "overshoots"
        or "undershoots" in the y-direction). In contrast, the widely used cubic spline
        'interpolation does not ensure monotonicity.
    show : bool
        If True, show a plot of the symmetry features.

    Returns
    -------
    pd.DataFrame
        A DataFrame of same length as :func:`.rsp_signal` containing the following columns:

        * ``"RSP_Symmetry_PeakTrough"``
        * ``"RSP_Symmetry_RiseDecay"``

    See Also
    --------
    rsp_clean, rsp_peaks, rsp_amplitude, rsp_phase

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      rsp = nk.rsp_simulate(duration=45, respiratory_rate=15)
      cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
      peak_signal, info = nk.rsp_peaks(cleaned)

      @savefig p_rsp_symmetry1.png scale=100%
      symmetry = nk.rsp_symmetry(cleaned, peak_signal, show=True)
      @suppress
      plt.close()

    References
    ----------
    * Cole, S., & Voytek, B. (2019). Cycle-by-cycle analysis of neural oscillations. Journal of
      neurophysiology, 122(2), 849-861.

    """
    # Format input.
    peaks, troughs = _rsp_fixpeaks_retrieve(peaks, troughs)

    # Sanity checks -----------------------------------------------------------
    failed_checks = False
    if len(peaks) <= 4 or len(troughs) <= 4:
        warn(
            "Not enough peaks and troughs (signal too short?) to compute symmetry"
            + ", returning nan for symmetry.",
            category=NeuroKitWarning,
        )
        failed_checks = True

    if np.any(peaks - troughs < 0):
        warn(
            "Peaks and troughs are not correctly aligned (i.e., not consecutive)"
            + ", returning nan for symmetry.",
            category=NeuroKitWarning,
        )
        failed_checks = True

    if failed_checks:
        return pd.DataFrame(
            {
                "RSP_Symmetry_PeakTrough": np.full(len(rsp_cleaned), np.nan),
                "RSP_Symmetry_RiseDecay": np.full(len(rsp_cleaned), np.nan),
            }
        )

    # Compute symmetry features -----------------------------------------------
    # See https://twitter.com/bradleyvoytek/status/1591495571269124096/photo/1

    # Rise-decay symmetry
    through_to_peak = peaks - troughs
    peak_to_through = troughs[1:] - peaks[:-1]
    risedecay_symmetry = through_to_peak[:-1] / (through_to_peak[:-1] + peak_to_through)

    # Find half-way points (trough to peak)
    halfway_values = (rsp_cleaned[peaks] - rsp_cleaned[troughs]) / 2
    halfway_values += rsp_cleaned[troughs]
    halfway_locations = np.zeros(len(halfway_values))
    for i in range(len(peaks)):
        segment = rsp_cleaned[troughs[i] : peaks[i]]
        halfway_locations[i] = (
            find_closest(halfway_values[i], segment, return_index=True) + troughs[i]
        )

    # Find half-way points (peak to next through)
    halfway_values2 = (rsp_cleaned[peaks[:-1]] - rsp_cleaned[troughs[1::]]) / 2
    halfway_values2 += rsp_cleaned[troughs[1::]]
    halfway_locations2 = np.zeros(len(halfway_values2))
    for i in range(len(peaks[:-1])):
        segment = rsp_cleaned[peaks[i] : troughs[i + 1]]
        halfway_locations2[i] = (
            find_closest(halfway_values2[i], segment, return_index=True) + peaks[i]
        )

    # Peak-trough symmetry
    asc_to_desc = halfway_locations2[1:] - halfway_locations[1:-1]
    desc_to_asc = halfway_locations[1:-1] - halfway_locations2[:-1]
    peaktrough_symmetry = desc_to_asc / (asc_to_desc + desc_to_asc)

    # Interpolate to length of rsp_cleaned.
    risedecay_symmetry = signal_interpolate(
        peaks[:-1],
        risedecay_symmetry,
        x_new=np.arange(len(rsp_cleaned)),
        method=interpolation_method,
    )
    peaktrough_symmetry = signal_interpolate(
        peaks[1:-1],
        peaktrough_symmetry,
        x_new=np.arange(len(rsp_cleaned)),
        method=interpolation_method,
    )

    if show is True:
        normalized = rescale(rsp_cleaned)  # Rescale to 0-1
        plt.plot(normalized, color="grey", label="Respiration (normalized)")
        plt.scatter(peaks, normalized[peaks], color="red")
        plt.scatter(troughs, normalized[troughs], color="blue")
        plt.scatter(halfway_locations, normalized[halfway_locations.astype(int)], color="orange")
        plt.scatter(
            halfway_locations2, normalized[halfway_locations2.astype(int)], color="darkgreen"
        )

        plt.plot(risedecay_symmetry, color="purple", label="Rise-decay symmetry")
        plt.plot(peaktrough_symmetry, color="green", label="Peak-trough symmetry")
        plt.legend()

    return pd.DataFrame(
        {
            "RSP_Symmetry_PeakTrough": peaktrough_symmetry,
            "RSP_Symmetry_RiseDecay": risedecay_symmetry,
        }
    )
