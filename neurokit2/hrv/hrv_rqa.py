# -*- coding: utf-8 -*-
import numpy as np
import scipy.spatial

from ..complexity import complexity_rqa
from ..signal import signal_detrend
from .hrv_utils import _hrv_get_rri, _hrv_sanitize_input


def hrv_rqa(
    peaks,
    sampling_rate=1000,
    dimension=7,
    delay=1,
    tolerance="zimatore2021",
    linelength=4,
    show=False
):
    """Recurrence quantification analysis (RQA) of Heart Rate Variability (HRV).

    RQA is a type of complexity analysis for non-linear systems (related to entropy and fractal dimensions).
    See the ``complexity_rqa()`` function for more information.

    This function is not run routinely as part of ``hrv_nonlinear()`` or ``hrv()`` because its
    main function, ``complexity_rqa()``, relies on the ``PyRQA`` package, which is not trivial to install.
    You will need to successfully install it first before being able to run ``hrv_rqa()``.

    Parameters
    ----------
    peaks : dict
        Samples at which cardiac extrema (i.e., R-peaks, systolic peaks) occur.
        Can be a list of indices or the output(s) of other functions such as ecg_peaks,
        ppg_peaks, ecg_process or bio_process.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks occur. Should be at
        least twice as high as the highest frequency in vhf. By default 1000.
    delay : int
        See ``complexity_rqa()`` for more information.
    dimension : int
        See ``complexity_rqa()`` for more information.
    tolerance : float
        See ``complexity_rqa()`` for more information. If 'zimatore2021', will be set to half of the
        mean pairwise distance between points.
    linelength : int
        See ``complexity_rqa()`` for more information.
    show : bool
        See ``complexity_rqa()`` for more information.

    See Also
    --------
    complexity_rqa, hrv_nonlinear

    Returns
    ----------
    rqa : float
         The RQA.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Download data
    >>> data = nk.data("bio_resting_5min_100hz")
    >>>
    >>> # Find peaks
    >>> peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)
    >>>
    >>> # Compute HRV indices
    >>> # hrv_rqa = nk.hrv_rqa(peaks, sampling_rate=100, show=True)

    References
    ----------
    - Zimatore, G., Falcioni, L., Gallotta, M. C., Bonavolontà, V., Campanella, M.,
    De Spirito, M., ... & Baldari, C. (2021). Recurrence quantification analysis of heart rate variability
    to detect both ventilatory thresholds. PloS one, 16(10), e0249504.
    - Ding, H., Crozier, S., & Wilson, S. (2008). Optimization of Euclidean distance threshold in the
    application of recurrence quantification analysis to heart rate variability studies. Chaos,
    Solitons & Fractals, 38(5), 1457-1467.

    """
    # Sanitize input
    peaks = _hrv_sanitize_input(peaks)
    if isinstance(peaks, tuple):  # Detect actual sampling rate
        peaks, sampling_rate = peaks[0], peaks[1]

    # Compute R-R intervals (also referred to as NN) in milliseconds
    rri, _ = _hrv_get_rri(peaks, sampling_rate=sampling_rate, interpolate=False)

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
        linelength=linelength,
        show=show,
    )

    return rqa
