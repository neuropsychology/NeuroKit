# -*- coding: utf-8 -*-
import numpy as np
import scipy.spatial

from ..complexity import complexity_rqa
from ..signal import signal_detrend
from .hrv_utils import _hrv_format_input


def hrv_rqa(
    peaks,
    sampling_rate=1000,
    dimension=7,
    delay=1,
    tolerance="zimatore2021",
    show=False,
    **kwargs,
):
    """**Recurrence Quantification Analysis (RQA) of Heart Rate Variability (HRV)**

    RQA is a type of complexity analysis used in non-linear dynamics (related to entropy and fractal
    dimensions). See :func:`.complexity_rqa` for more information.

    Parameters
    ----------
    peaks : dict
        Samples at which cardiac extrema (i.e., R-peaks, systolic peaks) occur.
        Can be a list of indices or the output(s) of other functions such as :func:`.ecg_peaks`,
        :func:`.ppg_peaks`, :func:`.ecg_process` or :func:`.bio_process`.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks occur. Should be at
        least twice as high as the highest frequency in vhf. By default 1000.
    delay : int
        See :func:`.complexity_rqa` for more information.
    dimension : int
        See :func:`.complexity_rqa` for more information.
    tolerance : float
        See :func:`.complexity_rqa` for more information. If ``"zimatore2021"``, will be set to half
        of the mean pairwise distance between points.
    show : bool
        See :func:`.complexity_rqa` for more information.
    **kwargs
        Other arguments to be passed to :func:`.complexity_rqa`.

    See Also
    --------
    complexity_rqa, hrv_nonlinear

    Returns
    ----------
    rqa : float
         The RQA.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Download data
      data = nk.data("bio_resting_5min_100hz")

      # Find peaks
      peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)

      # Compute HRV RQA indices
      @savefig p_hrv_rqa1.png scale=100%
      hrv_rqa = nk.hrv_rqa(peaks, sampling_rate=100, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      hrv_rqa

    References
    ----------
    * Zimatore, G., Falcioni, L., Gallotta, M. C., Bonavolontà, V., Campanella, M., De Spirito, M.,
      ... & Baldari, C. (2021). Recurrence quantification analysis of heart rate variability to
      detect both ventilatory thresholds. PloS one, 16(10), e0249504.
    * Ding, H., Crozier, S., & Wilson, S. (2008). Optimization of Euclidean distance threshold in
      the application of recurrence quantification analysis to heart rate variability studies.
      Chaos, Solitons & Fractals, 38(5), 1457-1467.

    """
    # Sanitize input
    # If given peaks, compute R-R intervals (also referred to as NN) in milliseconds
    rri, _, _ = _hrv_format_input(peaks, sampling_rate=sampling_rate)

    # Linear detrend (Zimatore, 2021)
    rri = signal_detrend(rri, method="polynomial", order=1)

    # Radius (50% of mean distance between all pairs of points in time)
    if tolerance == "zimatore2021":
        dists = scipy.spatial.distance.pdist(np.array([rri, rri]).T, "euclidean")
        tolerance = 0.5 * np.mean(dists)

    # Run the RQA
    rqa, _ = complexity_rqa(
        rri,
        dimension=dimension,
        delay=delay,
        tolerance=tolerance,
        show=show,
        **kwargs,
    )

    return rqa
