News
=====

0.3.1
-------------------
Compatibility
+++++++++++++

* Python 3.15 is now supported.
* ``setuptools`` is now only installed with the ``full`` extra, removing its ``<82`` pin.

Changes in Results
++++++++++++++++++

* ``fractal_correlation()`` with the default ``radius``, and therefore ``HRV_CD``, can differ.
* ``expspace()`` returns different values when ``start`` is above 0.
* ``entropy_approximate()`` (and ``entropy_fuzzy(approximate=True)``, ``entropy_range(approximate=True)``,
  ``entropy_multiscale(method="MSApEn")`` and the ``complexity_tolerance(method="maxApEn")`` curve)
  now returns the signed difference Phi(m) - Phi(m+1) as defined by Pincus (1991), instead of its
  absolute value. It can therefore be slightly negative when the embedded vectors match only
  themselves (small tolerance, high dimension or short signal), where the absolute value used to
  make this degenerate estimate look like a small positive entropy.
* ``entropy_sample()`` (and ``entropy_fuzzy()``, ``entropy_range()`` and ``entropy_multiscale()``
  with ``"MMSEn"`` or ``"IMSEn"``) can differ slightly when ``delay`` is above 1: the matches at
  dimension *m* are now counted among the same templates as at *m* + 1, as defined by Richman &
  Moorman (2000). Before, the last ``delay - 1`` vectors at *m* had no counterpart at *m* + 1.
  Results with ``delay=1`` are unchanged.
* ``complexity_lempelziv(permutation=True)`` (and ``entropy_multiscale(method="LZC", permutation=True)``)
  with ``dimension`` of 4 or more now gives the correct number of words. Before, the ordinal
  patterns were joined into a string of their indices, so patterns 10 and above were written with
  two digits and parsed as two symbols, which inflated the count and the sequence length used for
  the normalization. Binary LZC (the default) is unchanged.
* ``complexity_delay()`` with the autocorrelation methods (``"theiler1990"``, ``"casdagli1991"``,
  ``"rosenstein1993"``) returned the delay plus one: the autocorrelation at lag 0 was labelled as
  delay 1. They now return the first lag at which the autocorrelation is at or below the
  threshold (1/e, 0, and 1 - 1/e of its initial value), and they honour a custom sequence of
  delays (a list ``delay_max`` was ignored).
* ``complexity_delay(method="rosenstein1994")`` now returns the delay at which the slope of the
  average displacement has decreased to 40% of its initial value, as in the paper. It used to look
  for a slope of 40 degrees on an arbitrarily scaled curve.
* ``complexity_delay(method="gautama2003")`` now uses the number of delay vectors, rather than the
  length of the signal, in the description-length penalty of the entropy ratio.
* ``complexity_dimension(method="fnn")`` now returns the first dimension at which the proportion
  of false nearest neighbours drops below 2% (or within 1.85 times its minimum when it stays
  above). It used to accept the dimension before the drop when the proportion reached exactly 0
  afterwards (e.g., 2 instead of 3 for the Lorenz attractor). Its test II now uses the same
  (Euclidean) distance in both dimensions.
* ``complexity_dimension(method="afnn")`` with a custom list of dimensions returned the index of
  the optimal dimension plus one rather than the dimension itself.
* ``complexity_tolerance(method="chon2009")`` was wrong by a factor of 4 or more (the fourth root
  of N / 1000 was computed as a division by N / 4000), used the extrapolated coefficients instead
  of the published ones for dimensions 3 to 7, and was not scaled by the SD of the signal (the
  formula gives r in units of the SD, so the same tolerance was returned whatever the amplitude
  of the signal).
* ``complexity_optimize()`` now calls ``complexity_delay()``, ``complexity_dimension()`` and
  ``complexity_tolerance()`` instead of its own copies of them, which tested one delay less than
  ``delay_max`` and mapped ``"rosenstein1993"`` to the average displacement method.
* ``fractal_petrosian()`` now uses the number of samples as *N* (Petrosian, 1995) rather than the
  number of consecutive differences, which is one less with the default ``symbolize="C"`` (and
  ``"D"`` and ``"R"``). The change is small (about 1e-4 for 500 samples) but systematic.
* ``complexity_symbolize(method="C")`` (used by default in ``fractal_petrosian()``) no longer
  treats a zero difference as an increase: a flat segment takes the sign of the previous nonzero
  difference, so it is not counted as a sign change. Before, a flat segment within a decrease
  counted as two sign changes. This only affects signals with repeated consecutive values (e.g.,
  quantized or low-resolution recordings).
* ``fractal_dfa()`` with ``overlap=True`` (the default, and therefore the ``DFA`` and ``MFDFA``
  indices of ``hrv_nonlinear()`` and ``rsp_rrv()``) now includes the last window that fits in the
  signal. Before, the window starting at ``len(signal) - window`` was left out, so the last
  samples were never used (e.g., the last 25 of 2,000 samples with windows of 50). The effect on
  alpha is small (up to about 0.01) but systematic. Results with ``overlap=False`` are unchanged.
* ``fractal_higuchi()`` and ``complexity_k()`` now return NaN for signals with missing values.
  Before, the increments touching a missing value were silently left out of the curve lengths,
  while their normalization still counted them.
* The ``"Intercept"`` of ``fractal_higuchi()`` and the ``"Intercepts"`` of ``complexity_k()`` are
  now those of the fitted line (log L(k) against log(1/k)); their sign was flipped before.
* ``complexity_k()``, and therefore ``fractal_higuchi()`` with its default ``k_max="default"``,
  now selects the first k_max from which the HFD stays within 0.01 of its value for the next 10
  tested k_max values. Before, ``find_plateau()`` kept the bend of the HFD curve with the highest
  HFD among the 10% sharpest ones (and returned the k_max just before it): the selected k_max was
  erratic (e.g., from 2 to 118 for 10 white noises of 500 samples, now from 4 to 14) and grew with
  the length of the signal (a median of 122 for pink noise of 500 samples and 756 for 2,000, now
  23 and 19). A list of k_max values is now tested in increasing order. ``find_plateau()`` itself
  is unchanged.

New Features
+++++++++++++

* ``complexity_delay(method="kim1999")`` (C-C method) is rewritten: it now splits the signal into
  the disjoint subseries of the paper (they were embedded with the delay a second time) and counts
  the pairs with vectorised distances, which makes it usable (a few seconds instead of minutes).
* ``complexity_dimension()`` works on quantized signals (e.g., RR intervals in integer
  milliseconds): for ``"fnn"``, identical vectors are accepted as neighbours and judged by
  Kennel's second test (the first one, a ratio, is undefined for them; they used to be silently
  counted as true neighbours); for ``"afnn"``, they are skipped instead of turning the whole
  estimate into NaN. Both methods return NaN with a warning,
  instead of raising, when no dimension can be evaluated.
* ``fractal_correlation()`` gains a ``theiler`` argument (Theiler window): pairs of points closer
  than that number of samples in time are left out of the correlation sum, which removes the bias
  of the autocorrelation of oversampled signals (e.g., EEG) at small radii. Defaults to 0 (all
  pairs), so results are unchanged.

Fixes
+++++++++++++

* ``complexity_delay(method="fraser1986")`` no longer fails on a noise-free periodic signal when
  the tested delays reach its period (the default number of bins of ``mutual_information()``
  diverged for perfectly correlated variables).
* ``complexity_tolerance()`` accepts numpy integers as ``dimension`` (e.g., the output of
  ``complexity_dimension()``), which used to raise "method not recognized" for ``"nolds"``,
  ``"chon2009"`` and ``"makowski"``; plotting the ``"recurrence"`` and ``"neighbours"`` methods no
  longer converts the returned scores to percentages in place, and the plots show the 1% and 2%
  targets; ``show=True`` works with the ``"makowski"`` method; the docstring now lists all the
  methods and which ones use ``delay`` and ``dimension``.
* ``complexity_lempelziv()`` is about 10 times faster, no longer fails on a one-sample signal, and
  with ``permutation=True`` its ``info`` keeps the ``"Permutation"``, ``"Dimension"`` and
  ``"Delay"`` entries (they were overwritten by those of ``complexity_ordinalpatterns()``).
* ``fractal_katz()``'s docstring now describes the version computed (absolute differences between
  successive values, ignoring time) and no longer claims a 1-1.5 range: KFD is not bounded by 2 and
  grows with the length of the signal for noise.
* ``fractal_correlation()`` is about 3 times faster and uses half the memory: the distances are
  computed once per pair (``scipy.spatial.distance.pdist()``) and the correlation sums of all the
  radii are counted in one pass. The radii are returned sorted. Results are unchanged, except with
  ``radius="Corr_Dim"`` for signals with exactly repeated points, whose smallest nonzero distance
  (which sets the radii) was rounding noise.
* ``fractal_correlation()``'s docstring now cites Grassberger & Procaccia (1983) and Theiler
  (1986), and notes that the default radii can reach beyond the scaling region of an attractor.
* ``fractal_dfa()`` is 5 to 10 times faster: the polynomial trends of all the windows of a scale
  are computed with one matrix product instead of a loop over the windows.
* ``fractal_dfa()``'s docstring now describes alpha correctly (0.5 for white noise, below 0.5 for
  anti-correlations, 1.5 for a Brownian motion, equal to the Hurst exponent only for stationary
  signals), gives the right minimum of the default scales (10 samples, not 4), and cites Peng et
  al. (1994).
* ``fractal_petrosian()``'s docstring now gives the right default for ``symbolize`` (``"C"``, the
  sign of consecutive differences, not the mean).
* ``complexity_k()`` (and therefore ``fractal_higuchi(k_max="default")``) is several hundred times
  faster (e.g., 0.04 s instead of 19 s for 1,600 samples): the curve lengths are computed once
  instead of again for each tested k_max, and the least-squares fits of all the k_max values are
  done at once. The HFD values are unchanged.
* ``complexity_k()`` and ``fractal_higuchi()`` now accept lists and pandas Series (with any
  index), and ``complexity_k()`` accepts numpy integers as ``k_max`` (it raised an
  ``UnboundLocalError``). ``complexity_k()`` raises an error for k_max values that are not between
  2 and half the length of the signal, and ``fractal_higuchi()`` warns before returning NaN. The
  k_max returned by ``complexity_k()`` is an int, and its warning when no plateau is found no
  longer says that the optimal k_max is 2 or less.
* ``complexity_hjorth()``'s docstring now gives the right formulas of mobility and complexity (the
  one given for complexity was the mobility, and mobility was defined through complexity), and
  the units of mobility.
* ``signal_formatpeaks()`` now drops every missing index together with its value. Before, only the
  first one was dropped, so with two or more unrecovered SCRs ``eda_peaks()`` put ``SCR_RecoveryTime``
  values at the wrong recovery points.
* ``expspace()`` now spans the full range and no longer depends on the units of the values.
* ``fractal_correlation()`` now accepts a list of radiuses and returns 0 for constant signals.
* ``signal_resample()`` no longer sets the last sample to 0 for some signal lengths.
* ``ecg_simulate()`` no longer hangs for some settings.
* ``complexity_dimension(method="fnn")`` (and ``complexity_optimize(dimension_method="fnn")``)
  now return the dimension at which the proportion of false nearest neighbours meets the
  criterion, rather than its index in the tested sequence. The result was one too low with the
  default sequence (e.g., 2 instead of 3 for the Lorenz attractor) and arbitrary for a custom
  ``dimension_max`` sequence; the marker drawn with ``show=True`` was off accordingly.
* ``complexity_optimize(dimension_method="fnn", show=True)`` no longer raises a shape error when
  plotting the false nearest neighbour fractions.


0.3.0
-------------------
Compatibility
+++++++++++++

* pandas 3 is now supported (the ``pandas<3`` pin was removed). Python 3.10 to 3.14 are tested.

Changes in Results
++++++++++++++++++

Several fixes below change computed values compared to 0.2.x:

* ``HRV_PSS`` is now weighted by the number of RR differences in short segments (e.g., from
  0.74 to 0.55 on ``bio_resting_5min_100hz``) (#1214).
* ``RSA_PorgesBohrer`` is now computed on heart period and is positive, in ln(ms^2); it was
  previously negative (#1231).
* ``HRV_SDANN*`` and ``HRV_SDNNI*`` now include the last interval of each window and use the
  actual interval timestamps when available (#1213, #1215).
* ``HRV_TINN`` no longer returns 0 when the interval histogram has large counts (#1216).
* ``ecg_delineate(method="dwt")`` P and T peaks (and the onsets/offsets derived from them) can
  differ for signals in their original units (#1220).
* ``epochs_create(events=None)`` now returns the final complete block, i.e., often one more
  epoch (#1222).
* ``rsp_rvt()`` differs at sampling rates other than 1000 Hz (#1233).
* ``standardize(robust=True)`` on multi-column data now scales each column by its own MAD (#1229).
* ``fractal_dfa()`` with negative ``q`` can shift slightly (#1209).

New Features
+++++++++++++

* ``read_xdf()`` returns richer metadata: ``info`` now contains ``stream_names``,
  ``stream_types``, ``channel_names``, ``channel_counts``, ``durations`` and a per-stream
  ``streams`` list (name, type, channels, nominal and effective sampling rates, start/end
  times, uid, source id, hostname). Metadata is computed from the streams that actually
  contribute to the returned data.
* ``ecg_peaks()``, ``ecg_findpeaks()``, ``ecg_clean()`` and ``ecg_process()`` support
  ``method="elgendi2013"``, the two-moving-average QRS detector of Elgendi (2013). ``"elgendi"``
  still refers to the 2010 detector (#1202, #1232).

Fixes
+++++++++++++

* ``hrv_nonlinear()`` weights short acceleration/deceleration segments by their
  number of RR differences when calculating ``HRV_PSS`` (#1214).
* ``hrv_time()``: ``SDANN`` and ``SDNNI`` dropped the last RR interval of each window, and
  ignored the actual interval timestamps (e.g., recordings with gaps) (#1213, #1215).
* ``hrv_time()``: ``TINN`` returned 0 when every candidate triangle fit had a squared error
  above a fixed ceiling, which happens with large interval counts (#1216).
* ``rsp_rvt()`` passes ``sampling_rate`` to breath detection in the ``"power2020"`` and
  ``"birn2006"`` methods; at lower sampling rates many breaths were skipped (#1233).
* ``entropy_spectral()`` forwards ``sampling_rate`` to ``signal_psd()`` instead of raising a
  ``TypeError``, so ``min_frequency`` and ``max_frequency`` are applied in Hz of the actual
  signal (#1219).
* ``epochs_create()`` retains the final complete block during automatic chunking
  and aligns blocks with the data when relative epoch starts are nonzero (#1222).
* ``signal_interpolate(method="monotone_cubic")`` applies boundary fill values to every
  out-of-range sample, including irregular or unordered output grids (#1227).
* ``standardize(window=...)`` accepts named pandas columns and preserves their labels and
  sample index when performing rolling standardization (#1228).
* ``standardize(robust=True)`` scales each input column by its own MAD, so columns
  measured in different units are standardized independently (#1229).
* ``hrv_rsa()``: ``RSA_PorgesBohrer`` is now computed on heart period (ms) instead of heart
  rate (bpm), so it is in ln(ms^2) as documented and no longer comes out negative (#1230,
  #1231).
* ``ecg_process()``, ``ecg_delineate()``, ``ecg_phase()`` and ``ecg_quality()`` no longer
  crash when no R-peak is detected; they return NaN/empty outputs with a warning.
* ``read_acqknowledge()`` returns the sampling rate as an ``int`` (previously a NumPy float,
  which broke downstream functions), and the file-reading logic was updated for pandas 3.
* ``epochs_create()`` and ``signal_recompose()`` were updated for pandas 3 and scipy 1.18.
* ``signal_recompose()`` now applies ``keep_sd`` to reconstructed components instead of
  discarding the reconstruction and filtering the original components (#1221).
* ``eda_process()`` now forwards a custom ``amplitude_min`` to ``eda_peaks()``
  instead of hardcoding ``0.1`` (#1197, #1203).
* ``rsp_intervalrelated()`` computes inspiration/expiration durations from sample positions
  rather than index labels, fixing wrong durations for time-indexed epochs (#1196, #1201).
* ``hrv()`` and related functions accept ``(peaks, sampling_rate)`` tuples as well as the
  ``(signals, info)`` tuples returned by processing functions (#1205).
* ``signal_detrend(method="tarvainen2002")``: the second-difference operator was truncated at
  the end of the signal, leaving the last samples undetrended. The linear system is now solved
  with a sparse solver, which is also orders of magnitude faster for long signals (#1198).
* ``ecg_intervalrelated()``, ``rsp_intervalrelated()``, ``eda_intervalrelated()``,
  ``emg_intervalrelated()`` and ``eog_intervalrelated()`` no longer carry features over from
  a previous call (e.g., ``HRV_DFA_alpha2`` from a long recording reappearing in the results
  of a short one). ``ecg_intervalrelated()`` also returns HRV indices as floats instead of
  nested arrays (#1226).
* ``ecg_delineate(method="prominence")`` no longer crashes when an R-peak falls at the boundary
  of a beat segment (#1181).
* Microstates: ``method="ica"`` was routed to PCA and ``method="aahc"`` was not recognized;
  ``train="all"`` selected only the first ``n_channels`` samples; precomputed GFP passed to
  ``microstates_peaks()`` was ignored; EEG standardization was performed across channels
  instead of across time; MNE ``Epochs`` input failed; microstate labels and
  ``GEV_per_microstate`` were incorrectly remapped after reordering; ``criterion="cv"``
  raised a ``TypeError``; backfitting is now invariant to the scale of the maps; the first
  microstate lifetime was overcounted by one sample (#1199).
* ``hrv_nonlinear()`` docstring now points to :func:`hrv_rqa` (#1207).
* ``fractal_dfa()``: the variance threshold used to discard numerically-zero detrended
  segments is now relative to the largest segment variance at each scale instead of an
  absolute ``1e-8`` cutoff, restoring invariance of ``h(q)`` under constant rescaling of the
  signal (#1208). Note: results computed with negative ``q`` on physiological signals in
  their natural units may shift slightly; the new values are the more correct ones.
* ``ecg_delineate(method="dwt")``: the choice among candidate T and P peaks no longer depends on the
  units or gain of the signal (e.g., mV vs. uV). The amplitude term of the candidate score is now
  scale-free and weighted against the latency penalty (``amplitude_weight``), which also changes the
  detected peaks (and the onsets/offsets derived from them) for signals in their original units
  (#1210, #1220).
* ``epochs_create()`` applies baseline correction when ``epochs_start`` contains
  a list of per-event start times (#1223).

Maintenance
+++++++++++++

* Source distributions no longer ship the tests, docs and data folders.
* Broken image links in the README and documentation were fixed.

0.2.8
-------------------
New Features
+++++++++++++

* New feature `events_find()`: is now able to combine multiple digital input channels,
  retrieve events from the combined events channel and differentiate between the inputs that
  occur simultaneously.



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
