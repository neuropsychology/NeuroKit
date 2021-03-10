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
* Add `nk.find_outliers()` to identify outliers (abnormal values)
* Add utility function - `nk.check_type()` to return appropriate boolean values of input (integer, list, ndarray, pandas dataframe or pandas series)
* (experimental) Add error bars in the summary plot method to illustrate standard error of each bin

Fix Bugs
+++++++++++++

* Fix type of value in `signal_formatpeaks()` to ensure slice assignment is done on the same type

