# - * - coding: utf-8 - * -
import numpy as np
import scipy.signal

from .signal_formatpeaks import _signal_formatpeaks_sanitize


def signal_tidypeaksonsets(
    signal,
    peaks,
    onsets,
    method="Charlton2022",
    **kwargs,
):
    """**Tidy peaks and onsets**

    Tidy up PPG peaks and onsets to make them conform to the following rules:
    (i) No two points at the same time
    (ii) At least one local minimum between consecutive peaks
    (iii) At least one local maximum between consecutive onsets
    (iv) Alternates between onsets and peaks
    (v) Starts with onset, and ends with peak
    (vi) Same number of peaks and onsets

    Parameters
    ----------
    signal :  Union[list, np.array, pd.Series]
        The signal (i.e., a time series) that contains the peaks and onsets, in the form of a vector of values.
    peaks : list or array or DataFrame or Series or dict
        The samples at which the peaks occur.
    onsets : list or array or DataFrame or Series or dict
        The samples at which the onsets occur.
    method : str
        ``"Charlton2022"`` uses the approach used in the ppg-beats toolbox, Zenodo, https://doi.org/10.5281/zenodo.6037646
        This approach was designed for PPG signals.
    **kwargs
        Other keyword arguments.

    Returns
    -------
    peaks_clean : array
        The corrected peak locations.
    onsets_clean : array
        The corrected onset locations.

    See Also
    --------
    ppg_findpeaks, ppg_peaks

    References
    ----------
    * Charlton, P. H. et al. (2022). Detecting beats in the photoplethysmogram: benchmarking open-source algorithms.
      Physiological Measurement, 46, 035002, doi:10.1088/1361-6579/adb89e

    """
    # Format input
    peaks = _signal_formatpeaks_sanitize(peaks)
    onsets = _signal_formatpeaks_sanitize(onsets)

    # If method Kubios
    if method.lower() == "charlton2022":
        info, peaks_clean, onsets_clean = _signal_fixpeaks_charlton(
            signal, peaks, onsets, **kwargs
        )
    else:
        raise ValueError(
            "`method` not found. Must be one of the following: 'charlton2022'."
        )
        
    return info, peaks_clean, onsets_clean


# =============================================================================
# Methods
# =============================================================================

# =============================================================================
# Charlton (2022) - main function
# =============================================================================
def _signal_fixpeaks_charlton(
    signal,
    peaks,
    onsets,
):
    """Charlton 2022 method – enforces consistency between peaks and onsets.
    
    Tidy up peaks and onsets to make them conform to the following rules:
    (i) No two points at the same time
    (ii) At least one local minimum between consecutive peaks
    (iii) At least one local maximum between consecutive onsets
    (iv) Alternates between onsets and peaks
    (v) Starts with onset, and ends with peak
    (vi) Same number of peaks and onsets
    
    """

    # (i)  No two points at the same time
    peaks, onsets = _remove_repeated_peaks_and_onsets(peaks, onsets)

    # (ii) At least one local minimum between consecutive peaks
    peaks = _ensure_extremum_between(signal, peaks, other_extrema_type='pk')

    # (iii) At least one local maximum between consecutive onsets
    onsets = _ensure_extremum_between(signal, onsets, other_extrema_type='on')

    # (iv) Alternates between onsets and peaks
    # If there are two consecutive peaks, then insert an onset between them
    onsets, peaks = _insert_missing_extremum(signal, onsets, peaks, other_extrema_type='pk')
    # If there are two consecutive onsets, then insert a peak between them
    peaks, onsets = _insert_missing_extremum(signal, peaks, onsets, other_extrema_type='on')

    # (v) Starts with onset, and ends with peak
    peaks, onsets = _ensure_starts_with_onset_ends_with_peak(peaks, onsets)

    # (vi) same number of peaks and onsets
    peaks, onsets = _ensure_same_no_peaks_onsets(peaks, onsets)

    peaks_clean = peaks
    onsets_clean = onsets

    info = {
        "method": "charlton2022",
        "extra_peaks": [i for i in peaks if i not in peaks_clean],
        "extra_onsets": [i for i in onsets if i not in onsets_clean],
    }

    return info, peaks_clean, onsets_clean

# =============================================================================
# Charlton (2022) - helper functions
# =============================================================================
def _remove_repeated_peaks_and_onsets(peaks, onsets):

    # remove any repeated peaks (or onsets)
    peaks = np.unique(peaks)
    onsets = np.unique(onsets)

    # If there is a peak and onset at the same index, then remove them both
    repeated = np.intersect1d(peaks, onsets)
    peaks = np.setxor1d(peaks, repeated)
    onsets = np.setxor1d(onsets, repeated)
    return peaks, onsets


def _ensure_extremum_between(signal, other_extrema, other_extrema_type='pk'):

    # If there are two peaks (or onsets) without a local minimum (or maximum) between them,
    # then remove the lower (or higher) one
    
    if other_extrema_type == 'pk':
        extrema = scipy.signal.argrelmin(signal)[0]
    else:
        extrema = scipy.signal.argrelmax(signal)[0]

    other_extrema = list(other_extrema)

    finished = False
    while not finished:
        els_to_remove = []
        for i in range(len(other_extrema) - 1):
            rel_els = np.arange(other_extrema[i] + 1, other_extrema[i + 1])
            if not np.intersect1d(rel_els, extrema).size:
                other_extrema_vals = signal[other_extrema[i:i+2]]
                if other_extrema_type == 'pk':
                    el_to_remove = int(np.argmin(other_extrema_vals))
                else:
                    el_to_remove = int(np.argmax(other_extrema_vals))
                els_to_remove.append(i + el_to_remove)
        if not els_to_remove:
            finished = True
        else:
            for i in reversed(els_to_remove):  # remove in reverse to avoid shifting
                del other_extrema[i]
    return np.array(other_extrema)


def _insert_missing_extremum(signal, extrema, other_extrema, other_extrema_type='pk'):
    """
    If there are two consecutive extrema of one type (known as other_extrema),
    then insert an extremum of the second type (known as extrema) between them.
    """

    other_extrema_log = np.concatenate([
        np.ones(len(other_extrema), dtype=bool),
        np.zeros(len(extrema), dtype=bool)
    ])
    els = np.concatenate([other_extrema, extrema])
    order = np.argsort(els)
    els = els[order]
    other_extrema_log = other_extrema_log[order]

    bad_els = np.where(
        (np.diff(other_extrema_log) == 0) & (other_extrema_log[:-1])
    )[0]  # repeated other_extrema
    
    if len(bad_els) > 0:  # if there is a repeated other extrema
        for bad_el_no in range(len(bad_els)):  # cycle through each repeated other extrema
            curr_other_extrema = [els[bad_els[bad_el_no]], els[bad_els[bad_el_no] + 1]]
            start = curr_other_extrema[0]
            end = curr_other_extrema[1]

            # Create baseline to remove
            bw_to_remove = np.linspace(signal[start], signal[end], end - start + 1)

            # Detrend segment
            segment = signal[start:end+1] - bw_to_remove

            # Find extremum in detrended segment
            if other_extrema_type == 'pk':
                temp = np.argmin(segment)
            else:
                temp = np.argmax(segment)

            # check this hasn't just detected one of the other_extrema (which can happen with strong baseline wander)
            if temp == 0 or temp == (end - start):
                # then just remove the first peak
                other_extrema = other_extrema[other_extrema != start]
            else:
                curr_new_extrema = start + temp
                extrema = np.sort(np.append(extrema, curr_new_extrema))

    return extrema, other_extrema

def _ensure_starts_with_onset_ends_with_peak(peaks, onsets):
    """
    Make sure that the first onset is before the first peak, and the last peak is after the last onset
    """

    finished = False
    while not finished:
        if len(onsets) > 0 and len(peaks) > 0 and onsets[0] > peaks[0]:
            peaks = peaks[1:]
        else:
            finished = True

    finished = False
    while not finished:
        if len(peaks) > 0 and len(onsets) > 0 and peaks[-1] < onsets[-1]:
            onsets = onsets[:-1]
        else:
            finished = True

    return peaks, onsets

def _ensure_same_no_peaks_onsets(peaks, onsets):
    """
    NB: This doesn't quite ensure the same no of peaks and onsets, it only does it for a specific condition
    """

    # if no peaks (or onsets) were detected, then don't output any indices for either
    if len(peaks) == 0:
        onsets = []
    if len(onsets) == 0:
        peaks = []

    return peaks, onsets