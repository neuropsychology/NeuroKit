#def _ecg_wave_detector(ecg_cleaned, rpeaks=None, sampling_rate=1000):
#    """
#    - **Cardiac Cycle**: A typical ECG heartbeat consists of a P wave, a QRS complex and a T wave.The P wave represents the wave of depolarization that spreads from the SA-node throughout the atria. The QRS complex reflects the rapid depolarization of the right and left ventricles. Since the ventricles are the largest part of the heart, in terms of mass, the QRS complex usually has a much larger amplitude than the P-wave. The T wave represents the ventricular repolarization of the ventricles. On rare occasions, a U wave can be seen following the T wave. The U wave is believed to be related to the last remnants of ventricular repolarization.
#
#    Examples
#    ----------
#    >>> import neurokit2 as nk
#    >>> ecg_cleaned = nk.ecg_clean(nk.ecg_simulate(duration=5))
#    >>> _, rpeaks = nk.ecg_peaks(ecg_cleaned)
#    """
#    # Sanitize input
#    if rpeaks is None:
#        rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)["ECG_R_Peaks"]
#
#    if isinstance(rpeaks, dict):
#        rpeaks = rpeaks["ECG_R_Peaks"]
#
#
#    # Initialize
#    heartbeats = nk.epochs_create(ecg_cleaned, rpeaks, sampling_rate=sampling_rate, epochs_start=1, epochs_end=3)
#
#    for i in range(len(rpeaks)):
#        heartbeat = heartbeats[str(i+1)]
#
#        # Q
#
#
#    q_waves = []
#    p_waves = []
#    q_waves_starts = []
#    s_waves = []
#    t_waves = []
#    t_waves_starts = []
#    t_waves_ends = []
#    for index, rpeak in enumerate(rpeaks[:-3]):
#
#        try:
#            epoch_before = np.array(ecg)[int(rpeaks[index-1]):int(rpeak)]
#            epoch_before = epoch_before[int(len(epoch_before)/2):len(epoch_before)]
#            epoch_before = list(reversed(epoch_before))
#
#            q_wave_index = np.min(find_peaks(epoch_before))
#            q_wave = rpeak - q_wave_index
#            p_wave_index = q_wave_index + np.argmax(epoch_before[q_wave_index:])
#            p_wave = rpeak - p_wave_index
#
#            inter_pq = epoch_before[q_wave_index:p_wave_index]
#            inter_pq_derivative = np.gradient(inter_pq, 2)
#            q_start_index = find_closest_in_list(len(inter_pq_derivative)/2, find_peaks(inter_pq_derivative))
#            q_start = q_wave - q_start_index
#
#            q_waves.append(q_wave)
#            p_waves.append(p_wave)
#            q_waves_starts.append(q_start)
#        except ValueError:
#            pass
#        except IndexError:
#            pass
#
#        try:
#            epoch_after = np.array(ecg)[int(rpeak):int(rpeaks[index+1])]
#            epoch_after = epoch_after[0:int(len(epoch_after)/2)]
#
#            s_wave_index = np.min(find_peaks(epoch_after))
#            s_wave = rpeak + s_wave_index
#            t_wave_index = s_wave_index + np.argmax(epoch_after[s_wave_index:])
#            t_wave = rpeak + t_wave_index
#
#            inter_st = epoch_after[s_wave_index:t_wave_index]
#            inter_st_derivative = np.gradient(inter_st, 2)
#            t_start_index = find_closest_in_list(len(inter_st_derivative)/2, find_peaks(inter_st_derivative))
#            t_start = s_wave + t_start_index
#            t_end = np.min(find_peaks(epoch_after[t_wave_index:]))
#            t_end = t_wave + t_end
#
#            s_waves.append(s_wave)
#            t_waves.append(t_wave)
#            t_waves_starts.append(t_start)
#            t_waves_ends.append(t_end)
#        except ValueError:
#            pass
#        except IndexError:
#            pass
#
## pd.Series(epoch_before).plot()
##    t_waves = []
##    for index, rpeak in enumerate(rpeaks[0:-1]):
##
##        epoch = np.array(ecg)[int(rpeak):int(rpeaks[index+1])]
##        pd.Series(epoch).plot()
##
##        # T wave
##        middle = (rpeaks[index+1] - rpeak) / 2
##        quarter = middle/2
##
##        epoch = np.array(ecg)[int(rpeak+quarter):int(rpeak+middle)]
##
##        try:
##            t_wave = int(rpeak+quarter) + np.argmax(epoch)
##            t_waves.append(t_wave)
##        except ValueError:
##            pass
##
##    p_waves = []
##    for index, rpeak in enumerate(rpeaks[1:]):
##        index += 1
##        # Q wave
##        middle = (rpeak - rpeaks[index-1]) / 2
##        quarter = middle/2
##
##        epoch = np.array(ecg)[int(rpeak-middle):int(rpeak-quarter)]
##
##        try:
##            p_wave = int(rpeak-quarter) + np.argmax(epoch)
##            p_waves.append(p_wave)
##        except ValueError:
##            pass
##
##    q_waves = []
##    for index, p_wave in enumerate(p_waves):
##        epoch = np.array(ecg)[int(p_wave):int(rpeaks[rpeaks>p_wave][0])]
##
##        try:
##            q_wave = p_wave + np.argmin(epoch)
##            q_waves.append(q_wave)
##        except ValueError:
##            pass
##
##    # TODO: manage to find the begininng of the Q and the end of the T wave so we can extract the QT interval
#
#
#    ecg_waves = {"T_Waves": t_waves,
#                 "P_Waves": p_waves,
#                 "Q_Waves": q_waves,
#                 "S_Waves": s_waves,
#                 "Q_Waves_Onsets": q_waves_starts,
#                 "T_Waves_Onsets": t_waves_starts,
#                 "T_Waves_Ends": t_waves_ends}
#    return(ecg_waves)