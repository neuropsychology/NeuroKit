News
=====

0.2.4
-------------------
Fixes
+++++++++++++

* `eda_sympathetic()` has been reviewed: low-pass filter and resampling have been added to be in
  line with the original paper
* `eda_findpeaks()` using methods proposed in nabian2018 is reviewed and improved. Differentiation
  has been added before smoothing. Skin conductance response criteria have been revised based on
  the original paper.



0.2.1
-------------------
New Features
+++++++++++++

* Allow for input with NaNs and extrapolation in `signal_interpolate()`
* Add argument `method` in `find_outliers()`
* A lot (see https://github.com/neuropsychology/NeuroKit/pull/645)




0.2.0
-------------------
New Features
+++++++++++++

* Add new time-domain measures in `hrv_time()`: `Prc20NN`, `Prc80NN`, `MinNN`, and `MaxNN`





0.1.6
-------------------

Breaking Changes
+++++++++++++++++

* Argument `type` changed to `out` in `expspace()`


New Features
+++++++++++++

* Add new time-domain measures in `hrv_time()`: `Prc20NN`, `Prc80NN`, `MinNN`, and `MaxNN`
* Allow `fix_peaks()` to account for larger intervals

Fixes
+++++++++++++





0.1.5
-------------------

Breaking Changes
+++++++++++++++++

* Argument `r` changed to `radius` in `fractal_correlation()`
* Argument `r` changed to `tolerance` in entropy and complexity utility functions
* Argument `r_method` changed to `tolerance_method` in `complexity_optimize()`
* `complexity_lempelziv()`, `fractal_higuchi()`, `fractal_katz()`, `fractal_correlation()`, `fractal_dfa()`, `entropy_multiscale()`, `entropy_shannon()`, `entropy_approximate()`, `entropy_fuzzy()`, `entropy_sample()` now return a tuple consisting of the complexity index, and a dictionary comprising of the different parameters specific to the measure. For `fractal_katz()` and `entropy_shannon()`, the parameters dictionary is empty.
* Restructure `complexity` submodules with optimization files starting with `optim_*`, such as `optim_complexity_delay()`, `optim_complexity_dimension()`, `optim_complexity_k()`, `optim_complexity_optimize()`, and `optim_complexity_tolerance()`.
* `mutual_information()` moved from `stats` module to `complexity` module.

New Features
+++++++++++++

* Added various complexity indices: `complexity_hjorth()`, `complexity_hurst()`, `complexity_lyapunov()`, `complexity_rqa()`, `complexity_rr()`, `entropy_coalition()`, `entropy_permutation()`, `entropy_range()`, `entropy_spectral()`, `fractal_nld()`, `fractal_psdslope()`, `fractal_sda()`, `fractal_sevcik()`
* Added `mne_templateMRI()` as a helper to get MNE's template MRI.
* Added `eeg_source()` as a helper to perform source reconstruction.
* Added `eeg_source_extract()` to extract the activity from a brain region.
* Added `parallel_run()` in `misc` as a parallel processing utility function.
* Added `find_plateau()` in `misc` to find the point of plateau in an array of values.
* Added `write_csv()` in `data` to facilitate saving dataframes into multiple parts.
* Added more complexity-related functions, `entropy_cumulative_residual()`, `entropy_differential()`, `entropy_svd()`, `fractal_petrosian()`, and `information_fisher()`.
* Updates logic to find `kmax` in `fractal_higuchi()`
* Add RSP_Amplitude_Baseline in event-related analysis
* Add argument `add_firstsamples` in `mne_channel_extract()` to account for first sample attribute in mne raw objects
* Allow plotting of `mne.Epochs` in `epochs_plot()`
* Add `mne_crop()` to crop `mne` Raw objects with additional flexibility to specify first and last elements
* Plotting function in `eeg_badchannels()` to visualize overlay of individual EEG channels and highlighting of bad ones
* Add `eog_peaks()` as wrapper for `eog_findpeaks()`
* Allow `ecg_delineate()` to account for different heart rate


Fixes
+++++++++++++

* Ensure detected offset in `emg_activation()` is not beyond signal length
* Raise ValueError in `_hrv_sanitize_input()` if RRIs are detected instead of peaks
* Ensure that multifractal DFA indices returned by `fractal_mdfa()` is not Nan when array of slopes contains Nan (due to zero fluctuations)
* Documentation of respiration from peak/trough terminology to inhale/exhale onsets
* Change labelling in `rsp_plot()` from "inhalation peaks" and "exhalation troughs" to "peaks (exhalation onsets)" and "troughs (inhalation onsets)" respectively.
* Change RSP_Amplitude_Mean/Min/Max parameters to be corrected based on value closest to t=0 in event-related analysis, rather than using all pre-zero values.
* Have `rsp_rrv()` compute breath-to-breath intervals based on trough indices (inhalation onsets) rather than peak indices
* Compute `rsp_rate()` based on trough indices (rather than peak indices) in 'periods' method


0.1.4.1
-------------------

Fixes
+++++++++++++
* Adjust `kmax` parameter in `fractal_higuchi()` according to signal length as having `kmax` more than half of signal length leads to division by zero error
* Ensure that sanitization of input in `_hrv_dfa()` is done before windows for `DFA_alpha2` is computed
* `np.seterr` is added to `fractal_dfa()` to avoid returning division by zero warning which is an expected behaviour


0.1.4
-------------------

Breaking Changes
+++++++++++++++++

* `fractal_df()` now returns a dictionary of windows, fluctuations and the slope value (see documentation for more information. If `multifractal` is True, the dictionary additionally contains the parameters of the singularity spectrum (see `singularity_spectrum()` for more information)

New Features
+++++++++++++

* Add convenience function `intervals_to_peaks()` useful for RRI or BBI conversion to peak indices
* `hrv_nonlinear()` and `rrv_rsp()` now return the parameters of singularity spectrum for multifractal DFA analysis
* Add new complexity measures in `fractal_higuchi()`, `fractal_katz()` and `fractal_lempelziv()`
* Add new time-domain measures in `hrv_time()`: `SDANN` and `SDNNI`
* Add new non-linear measures in `hrv_nonlinear()`: `ShanEn`, `FuzzyEn`, `HFD`, `KFD` and `LZC`

Fixes
+++++++++++++

* Add path argument in `mne_data()` and throw warning to download mne datasets if data folder is not present
* The implementation of `TTIN` in `hrv_time()` is amended to its correct formulation.
* The default binsize used for RRI histogram in the computation of geometric HRV indices is set to 1 / 128 seconds


0.1.3
-------------------

Breaking Changes
+++++++++++++++++

* None

New Features
+++++++++++++

* Add internal function for detecting missing data points and forward filling missing values in `nk.*_clean()` functions
* Add computation of standard deviation in `eventrelated()` functions for *ECG_Rate_SD*, *EMG_Amplitude_SD*, *EOG_Rate_SD*, *PPG_Rate_SD*, *RSP_Rate_SD*, *RSP_Amplitude_SD*
* Add labelling for interval related features if a dictionary of dataframes is passed
* Retrun Q peaks and S Peaks information for wavelet-based methods in `nk.ecg_delineate()`

Fixes
+++++++++++++

* Fix epochs columns with `dtype: object` generated by `nk.epochs_create()`
* Bug fix ecg_findpeaks_rodrigues for array out of bounds bug


0.1.2
-------------------

New Features
+++++++++++++

* Additional features for `nk.rsp_intervalrelated()`: average inspiratory and expiratory durations, inspiratory-to-expiratory (I/E) time ratio
* Add multiscale entropy measures (MSE, CMSE, RCMSE) and fractal methods (Detrended Fluctuation Analysis, Correlation Dimension) into `nk.hrv_nonlinear()`
* Allow for data resampling in `nk.read_bitalino()`
* Add `bio_resting_8min_200hz` into database for reading with `nk.data()`
* Reading of url links in `nk.data()`
* Allow for `nk.hrv()` to compute RSA indices if respiratory data is present
* All `hrv` functions to automatically detect correct sampling rate if tuple or dict is passed as input
* Add support for PPG analysis: `nk.ppg_eventrelated()`, `nk.ppg_intervalrelated()`, `nk.ppg_analyze()`
* Add Zhao et al. (2018) method for `nk.ecg_quality()`
* Add tests for `epochs` module
* Add sub-epoch option for ECG and RSP event-related analysis:
	* users can create a smaller sub-epoch within the event-related epoch
	* the rate-related features of ECG and RSP signals are calculated over the sub-epoch
	* the remaining features are calculated over the original epoch, not the sub-epoch

Fixes
+++++++++++++

* Fix propagation of values in `nk.signal_formatpeaks()` for formatting SCR column outputs generated by `eda_peaks()`
* Fix docstrings of `nk.rsp_phase()`, from "RSP_Inspiration" to "RSP_Phase"
* Update `signal_filter()` method for `rsp_clean()`: to use `sos` form, instead of `ba` form of butterworth (similar to `eda_clean()`)





0.1.1
-------------------

New Features
+++++++++++++

* Use duration from `nk.events_find()` as `epochs_end` in `nk.epochs_create()`
* Allow customized subsets of epoch lengths in `nk.bio_analyze()` with `window_lengths` argument
* Add `nk.find_outliers()` to identify outliers (abnormal values)
* Add utility function - `nk.check_type()` to return appropriate boolean values of input (integer, list, ndarray, pandas dataframe or pandas series)
* (experimental) Add error bars in the summary plot method to illustrate standard error of each bin


Fixes
+++++++++++++

* Fix type of value in `nk.signal_formatpeaks()` to ensure slice assignment is done on the same type


0.0.1 (2019-10-29)
-------------------

* First release on PyPI.



