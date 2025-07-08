# - * - coding: utf-8 - * -
import numpy as np

from ..epochs import epochs_to_df
from ..signal import signal_interpolate, signal_cyclesegment


def signal_quality(
    signal, beat_inds, signal_type, sampling_rate=1000, method="templatematch"
):
    """**Assess quality of signal by comparing individual beat morphologies with a template**

    Assess the quality of a signal (e.g. PPG or ECG) using the specified method. You can pass an unfiltered
    signal as an input, but typically a filtered signal (e.g. cleaned using ``ppg_clean()`` or ``ecg_clean()``) will result in
    more reliable results. The following methods are available:

    * The ``"templatematch"`` method (loosely based on Orphanidou et al., 2015) computes a continuous
      index of quality of the PPG or ECG signal, by calculating the correlation coefficient between each
      individual beat's morphology and an average (template) beat morphology. This index is therefore
      relative: 1 corresponds to a signal where each individual beat's morphology is closest to the average beat morphology
      (i.e. correlate exactly with it) and 0 corresponds to there being no correlation with the average beat morphology.

    * The ``"disimilarity"`` method (loosely based on Sabeti et al., 2019) computes a continuous index
      of quality of the PPG or ECG signal, by calculating the level of disimilarity between each individual
      beat's morphology and an average (template) beat morpholoy (after they are normalised). A value of
      zero indicates no disimilarity (i.e. equivalent beat morphologies), whereas values above or below
      indicate increasing disimilarity. The original method used dynamic time-warping to align the pulse
      waves prior to calculating the level of dsimilarity, whereas this implementation does not currently
      include this step.


    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The cleaned signal, such as that returned by ``ppg_clean()`` or ``ecg_clean()``.
    beat_inds : tuple or list
        The list of beat samples (e.g. PPG or ECG peaks returned by ``ppg_peaks()`` or ``ecg_peaks()``).
    signal_type : str
        The signal type (e.g. 'ppg' or 'ecg').
    sampling_rate : int
        The sampling frequency of ``signal`` (in Hz, i.e., samples/second). Defaults to 1000.
    method : str
        The processing pipeline to apply. Can be one of ``"disimilarity"``, ``"templatematch"``. The default is
        ``"templatematch"``.
    **kwargs
        Additional keyword arguments, usually specific for each method.

    Returns
    -------
    quality : array
        Vector containing the quality index ranging from 0 to 1 for ``"templatematch"`` method,
        or an unbounded value (where 0 indicates high quality) for ``"disimilarity"`` method.

    See Also
    --------
    ppg_quality

    References
    ----------
    * Orphanidou, C. et al. (2015). "Signal-quality indices for the electrocardiogram and photoplethysmogram:
      derivation and applications to wireless monitoring". IEEE Journal of Biomedical and Health Informatics, 19(3), 832-8.
    * Sabeti E. et al. (2019). Signal quality measure for pulsatile physiological signals using morphological features:
      Applications in reliability measure for pulse oximetry. Informatics in Medicine Unlocked, 16, 100222.
    """

    signal_type = signal_type.lower()  # remove capitalised letters

    # Run selected quality assessment method
    if method in ["templatematch"]:  # Based on the approach in Orphanidou et al. (2015)
        quality = _quality_templatematch(
            signal,
            beat_inds=beat_inds,
            signal_type=signal_type,
            sampling_rate=sampling_rate,
        )
    elif method in ["disimilarity"]:  # Based on the approach in Sabeti et al. (2019)
        quality = _quality_disimilarity(
            signal,
            beat_inds=beat_inds,
            signal_type=signal_type,
            sampling_rate=sampling_rate,
        )

    return quality


# =============================================================================
# Calculate template morphology
# =============================================================================
def _calc_template_morph(signal, beat_inds, signal_type, sampling_rate=1000):

    # Segment to get individual beat morphologies
    heartbeats = signal_cyclesegment(signal, beat_inds, sampling_rate)

    # convert these to dataframe
    ind_morph = epochs_to_df(heartbeats).pivot(
        index="Label", columns="Time", values="Signal"
    )
    ind_morph.index = ind_morph.index.astype(int)
    ind_morph = ind_morph.sort_index()

    # Filter Nans
    valid_beats_mask = ~ind_morph.isnull().any(axis=1)
    ind_morph = ind_morph[valid_beats_mask]
    beat_inds = np.array(beat_inds)[valid_beats_mask.values]

    # Find template pulse wave as the average pulse wave shape
    templ_pw = ind_morph.mean()

    return templ_pw, ind_morph, beat_inds


# =============================================================================
# Quality assessment using template-matching method
# =============================================================================
def _quality_templatematch(
    signal, beat_inds=None, signal_type="ppg", sampling_rate=1000
):

    # Obtain individual beat morphologies and template beat morphology
    templ_morph, ind_morph, beat_inds = _calc_template_morph(
        signal,
        beat_inds=beat_inds,
        signal_type=signal_type,
        sampling_rate=sampling_rate,
    )

    # Find correlation coefficients (CCs) between individual beat morphologies and the template
    cc = np.zeros(len(beat_inds) - 1)
    for beat_no in range(0, len(beat_inds) - 1):
        temp = np.corrcoef(ind_morph.iloc[beat_no], templ_morph)
        cc[beat_no] = temp[0, 1]

    # Interpolate beat-by-beat CCs
    quality = signal_interpolate(
        beat_inds[0:-1], cc, x_new=np.arange(len(signal)), method="previous"
    )

    return quality


# =============================================================================
# Disimilarity measure method
# =============================================================================
def _norm_sum_one(pw):

    # ensure all values are positive
    pw = pw - pw.min() + 1

    # normalise pulse wave to sum to one
    pw = pw / np.sum(pw)

    return pw


def _calc_dis(pw1, pw2):
    # following the methodology in https://doi.org/10.1016/j.imu.2019.100222 (Sec. 3.1.2.5)

    # convert to numpy arrays
    pw1 = np.array(pw1)
    pw2 = np.array(pw2)

    # normalise to sum to one
    pw1 = _norm_sum_one(pw1)
    pw2 = _norm_sum_one(pw2)

    # ignore any elements which are zero because log(0) is -inf
    rel_els = (pw1 != 0) & (pw2 != 0)

    # calculate disimilarity measure (using pw2 as the template)
    dis = np.sum(pw2[rel_els] * np.log(pw2[rel_els] / pw1[rel_els]))

    return dis


# =============================================================================
# Quality assessment using disimilarity method
# =============================================================================
def _quality_disimilarity(
    signal, beat_inds=None, signal_type="ppg", sampling_rate=1000
):

    # Obtain individual beat morphologies and template beat morphology
    templ_morph, ind_morph, beat_inds = _calc_template_morph(
        signal,
        beat_inds=beat_inds,
        signal_type=signal_type,
        sampling_rate=sampling_rate,
    )

    # Find individual disimilarity measures
    dis = np.zeros(len(beat_inds) - 1)
    for beat_no in range(0, len(beat_inds) - 1):
        dis[beat_no] = _calc_dis(ind_morph.iloc[beat_no], templ_morph)

    # Interpolate beat-by-beat dis's
    quality = signal_interpolate(
        beat_inds[0:-1], dis, x_new=np.arange(len(signal)), method="previous"
    )

    return quality
