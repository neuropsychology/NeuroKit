# Load the NeuroKit package
import neurokit2 as nk

# Simulate 10 seconds of EDA signal (recorded at 250 samples / second)
eda_signal = nk.eda_simulate(duration=10, sampling_rate=250, n_scr=3, drift=0.01)

# Process the raw EDA signal
signals, info = nk.eda_process(eda_signal, sampling_rate=250)

# Extract clean EDA and SCR features
cleaned = signals["EDA_Clean"]
features = [info["SCR_Onsets"], info["SCR_Peaks"], info["SCR_Recovery"]]

# Visualize SCR features in cleaned EDA signal
plot = nk.events_plot(features, cleaned, color=['red', 'blue', 'orange'])
plot.savefig("edafeatures_1.png", dpi=300)

# Phasic and Tonic Components
data = nk.eda_phasic(nk.standardize(eda_signal), sampling_rate=250)
data["EDA_Raw"] = eda_signal
plot = data.plot()
plot.get_figure().savefig("edafeatures_2.png", dpi=300)

# Quick Plot
plot = nk.eda_plot(signals)
plot.savefig("edafeatures_3.png", dpi=300)
