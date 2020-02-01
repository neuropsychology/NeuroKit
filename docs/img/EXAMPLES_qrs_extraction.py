# Load the NeuroKit package
import neurokit2 as nk

# Simulate 30 seconds of ECG Signal (recorded at 250 samples / second)
ecg_signal = nk.ecg_simulate(duration=30, sampling_rate=250)

# Automatically process the (raw) ECG signal
signals, info = nk.ecg_process(ecg_signal, sampling_rate=250)

# Extract clean ECG and R-peasks location
rpeaks = info["ECG_R_Peaks"]
cleaned_ecg = signals["ECG_Clean"]

# Visualize R-peaks in ECG signal
nk.events_plot(rpeaks, cleaned_ecg)

# Segment the signal around the R-peaks
epochs = nk.epochs_create(cleaned_ecg, events=rpeaks, sampling_rate=250, epochs_start=-0.4, epochs_duration=1)

# Plotting all the heart beats
nk.epochs_plot(epochs)

nk.ecg_plot(signals)
