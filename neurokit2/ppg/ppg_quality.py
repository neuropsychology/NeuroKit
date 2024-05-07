# - * - coding: utf-8 - * -

import numpy as np

from ..epochs import epochs_to_df
from ..signal import signal_interpolate
from .ppg_peaks import ppg_peaks
from .ppg_segment import ppg_segment


def ppg_quality(
    ppg_cleaned, ppg_pw_peaks=None, sampling_rate=1000, method="templatematch", approach=None
):
    """**PPG Signal Quality Assessment**

    Assess the quality of the PPG Signal using various methods:

    * The ``"templatematch"`` method (loosely based on Orphanidou et al., 2015) computes a continuous
      index of quality of the PPG signal, by calculating the correlation coefficient between each
      individual pulse wave and an average (template) pulse wave shape. This index is therefore
      relative: 1 corresponds to pulse waves that are closest to the average pulse wave shape (i.e.
      correlate exactly with it) and 0 corresponds to there being no correlation with the average
      pulse wave shape. Note that 1 does not necessarily mean "good": use this index with care and
      plot it alongside your PPG signal to see if it makes sense.

    * The ``"disimilarity"`` method (loosely based on Sabeti et al., 2019) computes a continuous index
      of quality of the PPG signal, by calculating the level of disimilarity between each individual 
      pulse wave and an average (template) pulse wave shape (after they are normalised). A value of
      zero indicates no disimilarity (i.e. equivalent pulse wave shapes), whereas values above or below
      indicate increasing disimilarity. The original method used dynamic time-warping to align the pulse
      waves prior to calculating the level of dsimilarity, whereas this implementation does not currently
      include this step.

    Parameters
    ----------
    ppg_cleaned : Union[list, np.array, pd.Series]
        The cleaned PPG signal in the form of a vector of values.
    ppg_pw_peaks : tuple or list
        The list of PPG pulse wave peak samples returned by ``ppg_peaks()``. If None, peaks is computed from
        the signal input.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    method : str
        The method for computing PPG signal quality, can be ``"templatematch"`` (default).

    Returns
    -------
    array
        Vector containing the quality index ranging from 0 to 1 for ``"templatematch"`` method,
        or an unbounded value (where 0 indicates high quality) for ``"disimilarity"`` method.

    See Also
    --------
    ppg_segment

    References
    ----------
    * Orphanidou, C. et al. (2015). "Signal-quality indices for the electrocardiogram and photoplethysmogram:
      derivation and applications to wireless monitoring". IEEE Journal of Biomedical and Health Informatics, 19(3), 832-8.

    Examples
    --------
    * **Example 1:** 'templatematch' method

    .. ipython:: python

      import neurokit2 as nk

      ppg = nk.ppg_simulate(duration=30, sampling_rate=300, heart_rate=80)
      ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=300)
      quality = nk.ppg_quality(ppg_cleaned, sampling_rate=300, method="templatematch")

      @savefig p_ppg_quality.png scale=100%
      nk.signal_plot([ppg_cleaned, quality], standardize=True)
      @suppress
      plt.close()

    """

    method = method.lower()  # remove capitalised letters

    # Run selected quality assessment method
    if method in ["templatematch"]:
        quality = _ppg_quality_templatematch(
            ppg_cleaned, ppg_pw_peaks=ppg_pw_peaks, sampling_rate=sampling_rate
        )
    elif method in ["disimilarity"]:
        quality = _ppg_quality_disimilarity(
            ppg_cleaned, ppg_pw_peaks=ppg_pw_peaks, sampling_rate=sampling_rate
        )

    return quality

# =============================================================================
# Calculate a template pulse wave
# =============================================================================
def _calc_template_pw(ppg_cleaned, ppg_pw_peaks=None, sampling_rate=1000):

    # Sanitize inputs
    if ppg_pw_peaks is None:
        _, ppg_pw_peaks = ppg_peaks(ppg_cleaned, sampling_rate=sampling_rate)
        ppg_pw_peaks = ppg_pw_peaks["PPG_Peaks"]

    # Get heartbeats
    heartbeats = ppg_segment(ppg_cleaned, ppg_pw_peaks, sampling_rate)
    pw_data = epochs_to_df(heartbeats).pivot(
        index="Label", columns="Time", values="Signal"
    )
    pw_data.index = pw_data.index.astype(int)
    pw_data = pw_data.sort_index()

    # Filter Nans
    missing = pw_data.T.isnull().sum().values
    nonmissing = np.where(missing == 0)[0]
    pw_data = pw_data.iloc[nonmissing, :]

    # Find template pulse wave
    templ_pw = pw_data.mean()

    return templ_pw, pw_data, ppg_pw_peaks


# =============================================================================
# Template-matching method
# =============================================================================
def _ppg_quality_templatematch(ppg_cleaned, ppg_pw_peaks=None, sampling_rate=1000):

    # Obtain individual pulse waves and template pulse wave
    templ_pw, pw_data, ppg_pw_peaks = _calc_template_pw(
            ppg_cleaned, ppg_pw_peaks=ppg_pw_peaks, sampling_rate=sampling_rate
        )
    
    # Find individual correlation coefficients (CCs)
    cc = np.zeros(len(ppg_pw_peaks)-1)
    for beat_no in range(0,len(ppg_pw_peaks)-1):
        temp = np.corrcoef(pw_data.iloc[beat_no], templ_pw)
        cc[beat_no] = temp[0,1]

    # Interpolate beat-by-beat CCs
    quality = signal_interpolate(
        ppg_pw_peaks[0:-1], cc, x_new=np.arange(len(ppg_cleaned)), method="quadratic"
    )

    return quality

# =============================================================================
# Disimilarity measure method
# =============================================================================
def _norm_sum_one(pw):

    # ensure all values are positive
    pw = pw - pw.min() + 1

    # normalise pulse wave to sum to one
    pw = [x / sum(pw) for x in pw]

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


def _ppg_quality_disimilarity(ppg_cleaned, ppg_pw_peaks=None, sampling_rate=1000):

    # Obtain individual pulse waves and template pulse wave
    templ_pw, pw_data, ppg_pw_peaks = _calc_template_pw(
            ppg_cleaned, ppg_pw_peaks=ppg_pw_peaks, sampling_rate=sampling_rate
        )
    
    # Find individual disimilarity measures
    dis = np.zeros(len(ppg_pw_peaks)-1)
    for beat_no in range(0,len(ppg_pw_peaks)-1):
        dis[beat_no] = _calc_dis(pw_data.iloc[beat_no], templ_pw)

    # Interpolate beat-by-beat dis's
    quality = signal_interpolate(
        ppg_pw_peaks[0:-1], dis, x_new=np.arange(len(ppg_cleaned)), method="previous"
    )

    return quality