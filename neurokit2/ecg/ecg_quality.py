# - * - coding: utf-8 - * -
from warnings import warn

import numpy as np
import scipy

from ..epochs import epochs_to_df
from ..misc import NeuroKitWarning
from ..signal import signal_interpolate
from ..signal.signal_power import signal_power
from ..signal.signal_quality import signal_quality
from ..stats import distance, rescale
from .ecg_peaks import ecg_peaks
from .ecg_segment import ecg_segment


def ecg_quality(
    ecg_cleaned, rpeaks=None, sampling_rate=1000, method="averageQRS", approach=None
):
    """**ECG Signal Quality Assessment**

    Assess the quality of the ECG Signal using various methods:

    * The ``"templatematch"`` method (loosely based on Orphanidou et al., 2015) computes a continuous
      index of quality of the ECG signal, by calculating the correlation coefficient between each
      individual beat morphology and an average (template) beat morphology. This index is therefore
      relative: 1 corresponds to individual beats that are closest to the beat morphology (i.e.
      correlate exactly with it) and 0 corresponds to there being no correlation with the average
      beat morphology. For comparison, the ``"averageQRS"`` method forces the signal the quality to
      vary between 0 (lowest) and 1 (highest). Therefore, even in a high quality signal, some beats will
      have low values (e.g. 0), whereas others will have high values (e.g. 1). In contrast, ``"templatematch"``
      computes a quality metric determined by the average of the correlations between the template beat morphology
      and each individual beat's morphology. Therefore, it is possible that all beats exhibit high values (e.g. >0.95),
      indicative of consistent beat morphologies across the signal.

    * The ``"averageQRS"`` method computes a continuous index of quality of the ECG signal, by
      interpolating the distance of each QRS segment from the average QRS segment present in the *
      data. This index is therefore relative: 1 corresponds to heartbeats that are the closest to
      the average sample and 0 corresponds to the most distant heartbeat from that average sample.
      Note that 1 does not necessarily means "good": if the majority of samples are bad, then being
      close to the average will likely mean bad as well. Use this index with care and plot it
      alongside your ECG signal to see if it makes sense.

    * The ``"zhao2018"`` method (Zhao et al., 2018) extracts several signal quality indexes (SQIs):
      QRS wave power spectrum distribution pSQI, kurtosis kSQI, and baseline relative power basSQI.
      An additional R peak detection match qSQI was originally computed in the paper but left out
      in this algorithm. The indices were originally weighted with a ratio of [0.4, 0.4, 0.1, 0.1]
      to generate the final classification outcome, but because qSQI was dropped, the weights have
      been rearranged to [0.6, 0.2, 0.2] for pSQI, kSQI and basSQI respectively.

    Parameters
    ----------
    ecg_cleaned : Union[list, np.array, pd.Series]
        The cleaned ECG signal in the form of a vector of values.
    rpeaks : tuple or list
        The list of R-peak samples returned by ``ecg_peaks()``. If None, peaks is computed from
        the signal input.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    method : str
        The method for computing ECG signal quality, can be ``"averageQRS"`` (default) or ``"zhao2018"``.
    approach : str
        The data fusion approach as documented in Zhao et al. (2018). Can be ``"simple"``
        or ``"fuzzy"``. The former performs simple heuristic fusion of SQIs and the latter performs
        fuzzy comprehensive evaluation. If ``None`` (default), simple heuristic fusion is used.
    **kwargs
        Keyword arguments to be passed to ``signal_power()`` in the computation of basSQI and pSQI.

    Returns
    -------
    array or str
        Vector containing the quality index ranging from 0 to 1 for ``"averageQRS"`` method,
        returns string classification (``Unacceptable``, ``Barely acceptable`` or ``Excellent``)
        of the signal for ``"zhao2018"`` method.

    See Also
    --------
    ecg_segment, ecg_delineate, signal_quality

    References
    ----------
    * Zhao, Z., & Zhang, Y. (2018). "SQI quality evaluation mechanism of single-lead ECG signal
      based on simple heuristic fusion and fuzzy comprehensive evaluation". Frontiers in
      Physiology, 9, 727.
    * Orphanidou, C. et al. (2015). "Signal-quality indices for the electrocardiogram and photoplethysmogram:
      derivation and applications to wireless monitoring". IEEE Journal of Biomedical and Health Informatics, 19(3), 832-8.

    Examples
    --------
    * **Example 1:** 'averageQRS' method

    .. ipython:: python

      import neurokit2 as nk

      ecg = nk.ecg_simulate(duration=30, sampling_rate=300, noise=0.2)
      ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=300)
      quality = nk.ecg_quality(ecg_cleaned, sampling_rate=300)

      @savefig p_ecg_quality.png scale=100%
      nk.signal_plot([ecg_cleaned, quality], standardize=True)
      @suppress
      plt.close()

    * **Example 2:** Zhao et al. (2018) method

    .. ipython:: python

      nk.ecg_quality(ecg_cleaned,
                     sampling_rate=300,
                     method="zhao2018",
                     approach="fuzzy")

    """

    method = method.lower()  # remove capitalised letters

    # Run quality assessment algorithm
    if method in ["averageqrs"]:
        quality = _ecg_quality_averageQRS(
            ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate
        )
    elif method in ["zhao2018", "zhao", "SQI"]:
        if approach is None:
            approach = "simple"
        elif approach not in ["simple", "fuzzy"]:
            warn(
                "Please enter a relevant input if using method='zhao2018',"
                " 'simple' for simple heuristic fusion approach or"
                " 'fuzzy' for fuzzy comprehensive evaluation.",
                category=NeuroKitWarning,
            )

        quality = _ecg_quality_zhao2018(
            ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate, mode=approach
        )
    elif method in ["templatematch", "orphanidou2015"]:
        # Detect R peaks (if not done already)
        if rpeaks is None:
            _, rpeaks = ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
            rpeaks = rpeaks["ECG_R_Peaks"]
        # Assess quality using template matching
        quality = signal_quality(
            ecg_cleaned,
            beat_inds=rpeaks,
            signal_type="ecg",
            sampling_rate=sampling_rate,
            method="templatematch",
        )

    return quality


# =============================================================================
# Average QRS method
# =============================================================================
def _ecg_quality_averageQRS(ecg_cleaned, rpeaks=None, sampling_rate=1000):
    # Sanitize inputs
    if rpeaks is None:
        _, rpeaks = ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        rpeaks = rpeaks["ECG_R_Peaks"]

    # Get heartbeats
    heartbeats = ecg_segment(ecg_cleaned, rpeaks, sampling_rate)
    data = epochs_to_df(heartbeats).pivot(
        index="Label", columns="Time", values="Signal"
    )
    data.index = data.index.astype(int)
    data = data.sort_index()

    # Filter Nans
    missing = data.T.isnull().sum().values
    nonmissing = np.where(missing == 0)[0]

    data = data.iloc[nonmissing, :]

    # Compute distance
    dist = distance(data, method="mean")
    dist = rescale(np.abs(dist), to=[0, 1])
    dist = np.abs(dist - 1)  # So that 1 is top quality

    # Replace missing by 0
    quality = np.zeros(len(heartbeats))
    quality[nonmissing] = dist

    # Interpolate
    quality = signal_interpolate(
        rpeaks, quality, x_new=np.arange(len(ecg_cleaned)), method="previous"
    )

    return quality


# =============================================================================
# Zhao (2018) method
# =============================================================================
def _ecg_quality_zhao2018(
    ecg_cleaned,
    rpeaks=None,
    sampling_rate=1000,
    window=1024,
    kurtosis_method="fisher",
    mode="simple",
    **kwargs
):
    """Return ECG quality classification of based on Zhao et al. (2018),
    based on three indices: pSQI, kSQI, basSQI (qSQI not included here).

    If "Excellent", the ECG signal quality is good.
    If "Unacceptable", analyze the SQIs. If kSQI and basSQI are unqualified, it means that
    noise artefacts are present, and de-noising the signal is important before reevaluating the
    ECG signal quality. If pSQI (or qSQI, not included here) are unqualified, recollect ECG data.
    If "Barely acceptable", ECG quality assessment should be performed again to determine if the
    signal is excellent or unacceptable.

    Parameters
    ----------
    ecg_cleaned : Union[list, np.array, pd.Series]
        The cleaned ECG signal in the form of a vector of values.
    rpeaks : tuple or list
        The list of R-peak samples returned by `ecg_peaks()`. If None, peaks is computed from
        the signal input.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    window : int
        Length of each window in seconds. See `signal_psd()`.
    kurtosis_method : str
        Compute kurtosis (kSQI) based on "fisher" (default) or "pearson" definition.
    mode : str
        The data fusion approach as documented in Zhao et al. (2018). Can be "simple" (default)
        or "fuzzy". The former performs simple heuristic fusion of SQIs and the latter performs
        fuzzy comprehensive evaluation.
    **kwargs
        Keyword arguments to be passed to `signal_power()`.

    Returns
    -------
    str
        Quality classification.
    """

    # Sanitize inputs
    if rpeaks is None:
        _, rpeaks = ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        rpeaks = rpeaks["ECG_R_Peaks"]

    # Compute indexes
    kSQI = _ecg_quality_kSQI(ecg_cleaned, method=kurtosis_method)
    pSQI = _ecg_quality_pSQI(
        ecg_cleaned, sampling_rate=sampling_rate, window=window, **kwargs
    )
    basSQI = _ecg_quality_basSQI(
        ecg_cleaned, sampling_rate=sampling_rate, window=window, **kwargs
    )

    # Classify indices based on simple heuristic fusion
    if mode == "simple":
        # First stage rules (0 = unqualified, 1 = suspicious, 2 = optimal)

        # Get the maximum bpm
        if len(rpeaks) > 1:
            ecg_rate = 60000.0 / (1000.0 / sampling_rate * np.min(np.diff(rpeaks)))
        else:
            ecg_rate = 1

        # pSQI classification
        if ecg_rate < 130:
            l1, l2, l3 = 0.5, 0.8, 0.4
        else:
            l1, l2, l3 = 0.4, 0.7, 0.3

        if pSQI > l1 and pSQI < l2:
            pSQI_class = 2
        elif pSQI > l3 and pSQI < l1:
            pSQI_class = 1
        else:
            pSQI_class = 0

        # kSQI classification
        if kSQI > 5:
            kSQI_class = 2
        else:
            kSQI_class = 0

        # basSQI classification
        if basSQI >= 0.95:
            basSQI_class = 2
        elif basSQI < 0.9:
            basSQI_class = 0
        else:
            basSQI_class = 1

        class_matrix = np.array([pSQI_class, kSQI_class, basSQI_class])
        n_optimal = len(np.where(class_matrix == 2)[0])
        n_suspicious = len(np.where(class_matrix == 1)[0])
        n_unqualified = len(np.where(class_matrix == 0)[0])
        if n_unqualified >= 2 or (n_unqualified == 1 and n_suspicious == 2):
            return "Unacceptable"
        elif n_optimal >= 2 and n_unqualified == 0:
            return "Excellent"
        else:
            return "Barely acceptable"

    # Classify indices based on fuzzy comprehensive evaluation
    elif mode == "fuzzy":
        # *R1 left out because of lack of qSQI

        # pSQI
        # UpH
        if pSQI <= 0.25:
            UpH = 0
        elif pSQI >= 0.35:
            UpH = 1
        else:
            UpH = 0.1 * (pSQI - 0.25)

        # UpI
        if pSQI < 0.18:
            UpI = 0
        elif pSQI >= 0.32:
            UpI = 0
        elif pSQI >= 0.18 and pSQI < 0.22:
            UpI = 25 * (pSQI - 0.18)
        elif pSQI >= 0.22 and pSQI < 0.28:
            UpI = 1
        else:
            UpI = 25 * (0.32 - pSQI)

        # UpJ
        if pSQI < 0.15:
            UpJ = 1
        elif pSQI > 0.25:
            UpJ = 0
        else:
            UpJ = 0.1 * (0.25 - pSQI)

        # Get R2
        R2 = np.array([UpH, UpI, UpJ])

        # kSQI
        # Get R3
        if kSQI > 5:
            R3 = np.array([1, 0, 0])
        else:
            R3 = np.array([0, 0, 1])

        # basSQI
        # UbH
        if basSQI <= 90:
            UbH = 0
        elif basSQI >= 95:
            UbH = basSQI / 100.0
        else:
            UbH = 1.0 / (1 + (1 / np.power(0.8718 * (basSQI - 90), 2)))

        # UbJ
        if basSQI <= 85:
            UbJ = 1
        else:
            UbJ = 1.0 / (1 + np.power((basSQI - 85) / 5.0, 2))

        # UbI
        UbI = 1.0 / (1 + np.power((basSQI - 95) / 2.5, 2))

        # Get R4
        R4 = np.array([UbH, UbI, UbJ])

        # evaluation matrix R (remove R1 because of lack of qSQI)
        # R = np.vstack([R1, R2, R3, R4])
        R = np.vstack([R2, R3, R4])

        # weight vector W (remove first weight because of lack of qSQI)
        # W = np.array([0.4, 0.4, 0.1, 0.1])
        W = np.array([0.6, 0.2, 0.2])

        S = np.array(
            [np.sum((R[:, 0] * W)), np.sum((R[:, 1] * W)), np.sum((R[:, 2] * W))]
        )

        # classify
        V = np.sum(np.power(S, 2) * [1, 2, 3]) / np.sum(np.power(S, 2))

        if V < 1.5:
            return "Excellent"
        elif V >= 2.40:
            return "Unnacceptable"
        else:
            return "Barely acceptable"


def _ecg_quality_kSQI(ecg_cleaned, method="fisher"):
    """Return the kurtosis of the signal, with Fisher's or Pearson's method."""

    if method == "fisher":
        return scipy.stats.kurtosis(ecg_cleaned, fisher=True)
    elif method == "pearson":
        return scipy.stats.kurtosis(ecg_cleaned, fisher=False)


def _ecg_quality_pSQI(
    ecg_cleaned,
    sampling_rate=1000,
    window=1024,
    num_spectrum=[5, 15],
    dem_spectrum=[5, 40],
    **kwargs
):
    """Power Spectrum Distribution of QRS Wave."""

    psd = signal_power(
        ecg_cleaned,
        sampling_rate=sampling_rate,
        frequency_band=[num_spectrum, dem_spectrum],
        method="welch",
        normalize=False,
        window=window,
        **kwargs
    )

    num_power = psd.iloc[0, 0]
    dem_power = psd.iloc[0, 1]

    return num_power / dem_power


def _ecg_quality_basSQI(
    ecg_cleaned,
    sampling_rate=1000,
    window=1024,
    num_spectrum=[0, 1],
    dem_spectrum=[0, 40],
    **kwargs
):
    """Relative Power in the Baseline."""
    psd = signal_power(
        ecg_cleaned,
        sampling_rate=sampling_rate,
        frequency_band=[num_spectrum, dem_spectrum],
        method="welch",
        normalize=False,
        window=window,
        **kwargs
    )

    num_power = psd.iloc[0, 0]
    dem_power = psd.iloc[0, 1]

    return (1 - num_power) / dem_power
