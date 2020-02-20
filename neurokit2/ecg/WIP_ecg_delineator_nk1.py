import numpy as np

from ..signal import signal_findpeaks
from ..signal import signal_smooth
from ..epochs import epochs_create
from .ecg import ecg_peaks



def _ecg_delineator_derivative(ecg_cleaned, rpeaks=None, sampling_rate=1000):
    """
    - **Cardiac Cycle**: A typical ECG heartbeat consists of a P wave, a QRS complex and a T wave.The P wave represents the wave of depolarization that spreads from the SA-node throughout the atria. The QRS complex reflects the rapid depolarization of the right and left ventricles. Since the ventricles are the largest part of the heart, in terms of mass, the QRS complex usually has a much larger amplitude than the P-wave. The T wave represents the ventricular repolarization of the ventricles. On rare occasions, a U wave can be seen following the T wave. The U wave is believed to be related to the last remnants of ventricular repolarization.

    Examples
    ----------
    >>> import neurokit2 as nk
    >>> ecg_cleaned = nk.ecg_clean(nk.ecg_simulate(duration=5))
    >>> _, rpeaks = nk.ecg_peaks(ecg_cleaned)
    >>> waves = _ecg_delineator_derivative(ecg_cleaned, rpeaks)
    >>>
    >>> # Visualize the peaks of the waves
    >>> nk.events_plot([
            waves["ECG_P_Peaks"],
            waves["ECG_Q_Peaks"],
            waves["ECG_S_Peaks"],
            waves["ECG_T_Peaks"]],
            ecg_cleaned)
    >>>
    >>> # Visualize the onsets and offsets of the waves
    >>> nk.events_plot([
            waves["ECG_P_Onsets"],
            waves["ECG_T_Offsets"]],
            ecg_cleaned)
    """
    # Sanitize input
    if rpeaks is None:
        rpeaks = ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)["ECG_R_Peaks"]

    if isinstance(rpeaks, dict):
        rpeaks = rpeaks["ECG_R_Peaks"]

    # Initialize
    heartbeats = epochs_create(ecg_cleaned, rpeaks, sampling_rate=sampling_rate, epochs_start=-0.35, epochs_end=0.5)

    Q_list = []
    P_list = []
    S_list = []
    T_list = []

    P_onsets = []
    T_offsets = []

    for i, rpeak in enumerate(rpeaks):
        heartbeat = heartbeats[str(i+1)]

        # Get index of heartbeat
        R = heartbeat.index.get_loc(np.min(heartbeat.index.values[heartbeat.index.values > 0]))

        # Peaks ------
        # Q wave
        Q_index, Q = _ecg_delineator_derivative_Q(rpeak, heartbeat, R)
        Q_list.append(Q_index)

        # P wave
        P_index, P = _ecg_delineator_derivative_P(rpeak, heartbeat, R, Q)
        P_list.append(P_index)

        # S wave
        S_index, S = _ecg_delineator_derivative_S(rpeak, heartbeat, R)
        S_list.append(S_index)

        # T wave
        T_index, T = _ecg_delineator_derivative_T(rpeak, heartbeat, R, S)
        T_list.append(T_index)

        # Onsets/Offsets ------
        P_onsets.append(_ecg_delineator_derivative_P_onset(rpeak, heartbeat, R, P))
        T_offsets.append(_ecg_delineator_derivative_T_offset(rpeak, heartbeat, R, T))


    out = {"ECG_P_Peaks": P_list,
           "ECG_Q_Peaks": Q_list,
           "ECG_S_Peaks": S_list,
           "ECG_T_Peaks": T_list,
           "ECG_P_Onsets": P_onsets,
           "ECG_T_Offsets": T_offsets}
    return out




# =============================================================================
# Peaks
# =============================================================================
def _ecg_delineator_derivative_Q(rpeak, heartbeat, R):
    segment = heartbeat[:0]  # Select left hand side

    Q = signal_findpeaks(-1*segment["Signal"],
                            height_min=0.05 * (segment["Signal"].max() - segment["Signal"].min()))
    if len(Q["Peaks"]) == 0:
        return np.nan, None
    Q = Q["Peaks"][-1]  # Select most right-hand side
    from_R = R - Q  # Relative to R
    return rpeak - from_R, Q



def _ecg_delineator_derivative_P(rpeak, heartbeat, R, Q):
    if Q is None:
        return np.nan, None

    segment = heartbeat.iloc[:Q]  # Select left of Q wave
    P = signal_findpeaks(segment["Signal"],
                            height_min=0.05 * (segment["Signal"].max() - segment["Signal"].min()))
    if len(P["Peaks"]) == 0:
        return np.nan, None
    P = P["Peaks"][-1]  # Select most right-hand side
    from_R = R - P  # Relative to R
    return rpeak - from_R, P




def _ecg_delineator_derivative_S(rpeak, heartbeat, R):
    segment = heartbeat[0:]  # Select right hand side
    S = signal_findpeaks(-segment["Signal"],
                            height_min=0.05 * (segment["Signal"].max() - segment["Signal"].min()))
    if len(S["Peaks"]) == 0:
        return np.nan, None

    S = S["Peaks"][0]  # Select most left-hand side
    return rpeak + S, S



def _ecg_delineator_derivative_T(rpeak, heartbeat, R, S):
    if S is None:
        return np.nan, None

    segment = heartbeat.iloc[R + S:]  # Select right of S wave
    T = signal_findpeaks(segment["Signal"],
                            height_min=0.05 * (segment["Signal"].max() - segment["Signal"].min()))
    if len(T["Peaks"]) == 0:
        return np.nan, None

    T = S + T["Peaks"][0]  # Select most left-hand side
    return rpeak + T, T

# =============================================================================
# Offsets / Onsets
# =============================================================================

def _ecg_delineator_derivative_P_onset(rpeak, heartbeat, R, P):
    if P is None:
        return np.nan, None

    segment = heartbeat.iloc[:P]  # Select left of P wave
    signal = signal_smooth(segment["Signal"].values, size=R/10)
    signal = np.gradient(np.gradient(signal))
    P_onset = np.argmax(signal)

    from_R = R - P_onset  # Relative to R
    return rpeak - from_R



def _ecg_delineator_derivative_T_offset(rpeak, heartbeat, R, T):
    if T is None:
        return np.nan, None

    segment = heartbeat.iloc[R + T:]  # Select left of P wave
    signal = signal_smooth(segment["Signal"].values, size=R/10)
    signal = np.gradient(np.gradient(signal))
    T_offset = np.argmax(signal)

    return rpeak + T + T_offset
