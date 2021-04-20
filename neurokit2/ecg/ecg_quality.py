# - * - coding: utf-8 - * -
import numpy as np
import scipy

from ..epochs import epochs_to_df
from ..signal.signal_power import signal_power
from ..signal import signal_interpolate
from ..stats import distance, rescale
from .ecg_peaks import ecg_peaks
from .ecg_segment import ecg_segment


def ecg_quality(ecg_cleaned, rpeaks=None, sampling_rate=1000, method="averageQRS"):
    """Quality of ECG Signal.

    The "averageQRS" method computes a continuous index of quality of the ECG signal, by interpolating the distance
    of each QRS segment from the average QRS segment present in the data. This index is
    therefore relative, and 1 corresponds to heartbeats that are the closest to the average
    sample and 0 corresponds to the most distance heartbeat, from that average sample.

    The approach by Zhao et la. (2018) was originally designed for signal with a length of 10 seconds.

    Returns
    -------
    array
        Vector containing the quality index ranging from 0 to 1.

    See Also
    --------
    ecg_segment, ecg_delineate

    References
    ----------
    - Zhao, Z., & Zhang, Y. (2018). "SQI quality evaluation mechanism of single-lead ECG signal based
      on simple heuristic fusion and fuzzy comprehensive evaluation". Frontiers in Physiology, 9, 727.

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> ecg = nk.ecg_simulate(duration=30, sampling_rate=300, noise=0.2)
    >>> ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=300)
    >>> quality = nk.ecg_quality(ecg_cleaned, sampling_rate=300)
    >>>
    >>> nk.signal_plot([ecg_cleaned, quality], standardize=True)

    """
    method = method.lower()  # remove capitalised letters
    # Run peak detection algorithm
    if method in ["averageqrs"]:
        quality = _ecg_quality_averageQRS(ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate)

    return quality


# =============================================================================
# Average QRS
# =============================================================================
def _ecg_quality_averageQRS(ecg_cleaned, rpeaks=None, sampling_rate=1000):

    # Sanitize inputs
    if rpeaks is None:
        _, rpeaks = ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        rpeaks = rpeaks["ECG_R_Peaks"]

    # Get heartbeats
    heartbeats = ecg_segment(ecg_cleaned, rpeaks, sampling_rate)
    data = epochs_to_df(heartbeats).pivot(index="Label", columns="Time", values="Signal")
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
    quality = signal_interpolate(rpeaks, quality, x_new=np.arange(len(ecg_cleaned)), method="quadratic")

    return quality


#=============================================================================
#Zhao (2018)
#=============================================================================
def _ecg_quality_kSQI(ecg_cleaned, method="Fisher"):
    """ Return the kurtosis of the signal, with Fisher's or Pearson's method.
    """

    if method == "Fisher":
        kurtosis = scipy.stats.kurtosis(ecg_cleaned, fisher=True)
    elif method == "Pearson":
        kurtosis = scipy.stats.kurtosis(ecg_cleaned, fisher=False)

    return kurtosis

def _ecg_quality_pSQI(ecg_cleaned, sampling_rate=1000, nseg=1024, num_spectrum=[5, 15], dem_spectrum=[5, 40], **kwargs):
    """Power Spectrum Distribution of QRS Wave.
    """

    psd = signal_power(ecg_cleaned, sampling_rate=sampling_rate,
                       frequency_band=[num_spectrum, dem_spectrum],
                       method="welch", normalize=False, window=nseg, **kwargs)
    
    num_power = psd.iloc[0][0]
    dem_power = psd.iloc[0][1]
    
    return num_power / dem_power

def _ecg_quality_basSQI(ecg_cleaned, sampling_rate=1000, nseg=1024, num_spectrum=[0, 1], dem_spectrum=[0, 40], **kwargs):
    """Relative Power in the Baseline.
    """
    psd = signal_power(ecg_cleaned, sampling_rate=sampling_rate,
                       frequency_band=[num_spectrum, dem_spectrum],
                       method="welch", normalize=False, window=nseg, **kwargs)
    
    num_power = psd.iloc[0][0]
    dem_power = psd.iloc[0][1]
    
    return 1 - num_power / dem_power


def _ecg_quality_zhao2018(signal, detector_1, detector_2, sampling_rate=1000, search_window=100, nseg=1024, mode='simple'):
    import numpy as np
    """Implemented by @TiagoTostas

    Parameters
    ----------
    signal : array
        Input ECG signal in mV.
    detector_1 : array
        Input of the first R peak detector.
    detector_2 : array
        Input of the second R peak detector.
    fs : int, float, optional
        Sampling frequency (Hz).
    search_window : int, optional
        Search window around each peak, in ms.
    nseg : int, optional
        Frequency axis resolution.
    mode : str, optional
        If 'simple', simple heuristic. If 'fuzzy', employ a fuzzy classifier.

    Returns
    -------
    str
        Quality classification.

    """

    if (len(detector_1) == 0 or len(detector_2) == 0):
        return 'Unacceptable'



    ## compute indexes
    qsqi = bSQI(detector_1, detector_2, fs=sampling_rate, mode='matching', search_window=search_window)
    psqi = fSQI(signal, fs=sampling_rate, nseg=nseg, num_spectrum=[5, 15], dem_spectrum=[5, 40])
    ksqi = kSQI(signal)
    bassqi = fSQI(signal, fs=sampling_rate, nseg=nseg, num_spectrum=[0, 1], dem_spectrum=[0, 40], mode='bas')

    if mode == 'simple':
        ## First stage rules (0 = unqualified, 1 = suspicious, 2 = optimal)
        ## qSQI rules
        # if qsqi > 0.90:
        #     qsqi_class = 2
        # elif qsqi < 0.60:
        #     qsqi_class = 0
        # else:
        #     qsqi_class = 1

        ## pSQI rules
        ## Get the maximum bpm
        if (len(detector_1) > 1):
            RR_max = 60000.0 / (1000.0 / sampling_rate * np.min(np.diff(detector_1)))
        else:
            RR_max = 1

        if RR_max < 130:
            l1, l2, l3 = 0.5, 0.8, 0.4
        else:
            l1, l2, l3 = 0.4, 0.7, 0.3

        if psqi > l1 and psqi < l2:
            pSQI_class = 2
        elif psqi > l3 and psqi < l1:
            pSQI_class = 1
        else:
            pSQI_class = 0

        ## kSQI rules
        if ksqi > 5:
            kSQI_class = 2
        else:
            kSQI_class = 0

        ## basSQI rules
        if bassqi >= 0.95:
            basSQI_class = 2
        elif bassqi < 0.9:
            basSQI_class = 0
        else:
            basSQI_class = 1

        class_matrix = np.array([pSQI_class, kSQI_class, basSQI_class])
        n_optimal = len(np.where(class_matrix == 2)[0])
        n_suspics = len(np.where(class_matrix == 1)[0])
        n_unqualy = len(np.where(class_matrix == 0)[0])
        if n_unqualy == 2 or (n_unqualy == 1 and n_suspics == 2):
            return 'Unacceptable'
        elif n_optimal >= 2 and n_unqualy == 0:
            return 'Excellent'
        else:
            return 'Barely acceptable'

    elif mode == 'fuzzy':
        # Transform qSQI range from [0, 1] to [0, 100]
        qsqi = qsqi * 100.0
        # UqH (Excellent)
        if qsqi <= 80:
            UqH = 0
        elif qsqi >= 90:
            UqH = qsqi / 100.0
        else:
            UqH = 1.0 / (1 + (1 / np.power(0.3 * (qsqi - 80), 2)))

        # UqI (Barely acceptable)
        UqI = 1.0 / (1 + np.power((qsqi - 75) / 7.5, 2))

        # UqJ (unacceptable)
        if qsqi <= 55:
            UqJ = 1
        else:
            UqJ = 1.0 / (1 + np.power((qsqi - 55) / 5.0, 2))

        # Get R1
        R1 = np.array([UqH, UqI, UqJ])

        # pSQI
        # UpH
        if psqi <= 0.25:
            UpH = 0
        elif psqi >= 0.35:
            UpH = 1
        else:
            UpH = 0.1 * (psqi - 0.25)

        # UpI
        if psqi < 0.18:
            UpI = 0
        elif psqi >= 0.32:
            UpI = 0
        elif psqi >= 0.18 and psqi < 0.22:
            UpI = 25 * (psqi - 0.18)
        elif psqi >= 0.22 and psqi < 0.28:
            UpI = 1
        else:
            UpI = 25 * (0.32 - psqi)

        # UpJ
        if psqi < 0.15:
            UpJ = 1
        elif psqi > 0.25:
            UpJ = 0
        else:
            UpJ = 0.1 * (0.25 - psqi)

        # Get R2
        R2 = np.array([UpH, UpI, UpJ])

        # kSQI
        # Get R3
        if ksqi > 5:
            R3 = np.array([1, 0, 0])
        else:
            R3 = np.array([0, 0, 1])

        # basSQI
        # UbH
        if bassqi <= 90:
            UbH = 0
        elif bassqi >= 95:
            UbH = bassqi / 100.0
        else:
            UbH = 1.0 / (1 + (1 / np.power(0.8718 * (bassqi - 90), 2)))

        # UbI
        if bassqi <= 85:
            UbI = 1
        else:
            UbI = 1.0 / (1 + np.power((bassqi - 85) / 5.0, 2))

        # UbJ
        UbJ = 1.0 / (1 + np.power((bassqi - 95) / 2.5, 2))

        # R4
        R4 = np.array([UbH, UbI, UbJ])

        # evaluation matrix R
        R = np.vstack([R1, R2, R3, R4])

        # weight vector W
        W = np.array([0.4, 0.4, 0.1, 0.1])

        S = np.array([np.sum((R[:, 0] * W)), np.sum((R[:, 1] * W)), np.sum((R[:, 2] * W))])

        # classify
        V = np.sum(np.power(S, 2) * [1, 2, 3]) / np.sum(np.power(S, 2))

        if (V < 1.5):
            return 'Excellent'
        elif (V >= 2.40):
            return 'Unnacceptable'
        else:
            return 'Barely acceptable'
