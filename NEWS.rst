News
=====


0.0.1 (2019-10-29)
-------------------

* First release on PyPI.

0.1.1
-------------------

New Features
+++++++++++++

* Use duration from `nk.events_find()` as `epochs_end` in `nk.epochs_create()`
* Allow customized subsets of epoch lengths in `nk.bio_analyze()` with `window_lengths` argument
* Add `nk.find_outliers()` to identify outliers (abnormal values)
* Add utility function - `nk.check_type()` to return appropriate boolean values of input (integer, list, ndarray, pandas dataframe or pandas series)
* (experimental) Add error bars in the summary plot method to illustrate standard error of each bin
* Additional features for `nk.rsp_intervalrelated()`: average inspiratory and expiratory durations, inspiratory-to-expiratory (I/E) time ratio
* Add multiscale entropy measures (MSE, CMSE, RCMSE) into `nk.hrv_nonlinear()` 
* Allow for data resampling in `nk.read_bitalino()`
* Add `bio_resting_8min_200hz` into database for reading with `nk.data()`
* Allow for `hrv()` to compute RSA indices if respiratory data is present
* All `hrv` functions to automatically detect correct sampling rate if tuple or dict is passed as input

Fix Bugs
+++++++++++++

* Fix type of value in `nk.signal_formatpeaks()` to ensure slice assignment is done on the same type
* Fix docstrings of `nk.rsp_phase()`, from "RSP_Inspiration" to "RSP_Phase"

