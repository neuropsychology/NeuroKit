import numpy as np
import pandas as pd
import pytest

import neurokit2 as nk
from neurokit2.signal.signal_formatpeaks import _signal_from_indices, _signal_sanitize_indices


def test_find_closest():
    vals = [3, 5, 6, 1, 2]
    assert nk.find_closest(1.8, vals) == 2
    assert nk.find_closest(1.8, vals, return_index=True) == 4
    np.testing.assert_array_equal(nk.find_closest([1.8, 3.6], vals), [2, 3])
    np.testing.assert_array_equal(nk.find_closest([1.8, 3.6], vals, return_index=True), [4, 0])

    # Tie-breaking must return the earliest matching index in `list_to_search_in`
    ties = [5, 3, 5, 1, 3]
    assert nk.find_closest(4.0, ties, return_index=True) == 0
    assert nk.find_closest(4.0, ties) == 5
    assert nk.find_closest(3.0, ties, return_index=True) == 1

    # Directions and `strictly`
    assert nk.find_closest(3, vals, direction="smaller", strictly=False) == 3
    assert nk.find_closest(3, vals, direction="smaller", strictly=True) == 2
    assert nk.find_closest(3, vals, direction="greater", strictly=False) == 3
    assert nk.find_closest(3, vals, direction="greater", strictly=True) == 5
    assert nk.find_closest(3, vals, direction="both", strictly=True) == 2

    # Unmatched bounds, NaNs, empty targets, and invalid direction
    assert np.isnan(nk.find_closest(0, vals, direction="smaller", strictly=True))
    assert np.isnan(nk.find_closest(0, vals, direction="smaller", strictly=True, return_index=True))
    res_mixed = nk.find_closest([0.0, 2.5, np.nan], vals, direction="smaller", strictly=False)
    assert np.isnan(res_mixed[0]) and res_mixed[1] == 2.0 and np.isnan(res_mixed[2])
    idx_mixed = nk.find_closest([0.0, 2.5, np.nan], vals, direction="smaller", strictly=False, return_index=True)
    assert np.isnan(idx_mixed[0]) and idx_mixed[1] == 4.0 and np.isnan(idx_mixed[2])

    assert np.isnan(nk.find_closest(1.0, [np.nan, np.nan]))
    assert np.all(np.isnan(nk.find_closest([1.0, 2.0], [np.nan])))
    assert len(nk.find_closest([], vals)) == 0

    with pytest.raises(ValueError, match="direction"):
        nk.find_closest(1.0, vals, direction="sideways")


def test_find_outliers():
    data = [-12, 2, 1, 3, 66.6, 2, 1, 3, 2, -42, 2, 4, 1, 12]

    # "standardize" and "z" must behave identically to "sd" rather than falling into "norm" ppf
    out_sd = nk.find_outliers(data, exclude=2, side="both", method="sd")
    out_std = nk.find_outliers(data, exclude=2, side="both", method="standardize")
    out_z = nk.find_outliers(data, exclude=2, side="both", method="z")
    np.testing.assert_array_equal(out_std, out_sd)
    np.testing.assert_array_equal(out_z, out_sd)
    assert np.sum(out_std) < len(data)
    np.testing.assert_array_equal(nk.find_outliers(data, exclude=(-2, 2), method="z"), out_sd)

    # "norm" and one-sided / NaN-tolerant "percentile"
    out_norm = nk.find_outliers(data, exclude=0.05, side="both", method="norm")
    assert out_norm[4] and out_norm[9]

    arr_nan = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0, np.nan])
    out_pct_both = nk.find_outliers(arr_nan, exclude=0.2, side="both", method="percentile")
    out_pct_right = nk.find_outliers(arr_nan, exclude=0.1, side="right", method="percentile")
    out_pct_left = nk.find_outliers(arr_nan, exclude=0.1, side="left", method="percentile")
    out_pct_tuple = nk.find_outliers(arr_nan, exclude=(0.1, 0.9), side="both", method="percentile")
    assert out_pct_both[0] and out_pct_both[10] and not out_pct_both[11]
    assert out_pct_right[10] and not out_pct_right[0]
    assert out_pct_left[0] and not out_pct_left[10]
    np.testing.assert_array_equal(out_pct_tuple, out_pct_both)

    with pytest.raises(ValueError, match="side"):
        nk.find_outliers(data, side="top")
    with pytest.raises(ValueError, match="method"):
        nk.find_outliers(data, method="iqr_invalid")


def test_signal_findpeaks_list_and_empty_peaks():
    sig = nk.signal_simulate(duration=3, frequency=2, sampling_rate=100)
    info_arr = nk.signal_findpeaks(sig)
    info_list = nk.signal_findpeaks(sig.tolist())
    info_ser = nk.signal_findpeaks(pd.Series(sig))

    for key in ("Peaks", "Distance", "Height", "Width", "Onsets", "Offsets"):
        np.testing.assert_allclose(info_list[key], info_arr[key], equal_nan=True)
        np.testing.assert_allclose(info_ser[key], info_arr[key], equal_nan=True)
    assert np.any(~np.isnan(info_list["Onsets"]))
    assert np.any(~np.isnan(info_list["Offsets"]))
    info_rel = nk.signal_findpeaks(sig, relative_height_min=0.2, relative_max=True)
    assert len(info_rel["Peaks"]) > 0

    # Flat signal / empty filtered peaks must not crash relative_max / relative_mean filtering
    flat_info = nk.signal_findpeaks([0.0, 0.0, 0.0, 0.0], height_min=1.0, relative_height_min=0.5, relative_max=True)
    assert len(flat_info["Peaks"]) == 0


def test_signal_formatpeaks_default_peak_indices_and_multiple_nan_columns():
    # Default peak_indices=None must extract peak index values (not key name strings)
    sig = nk.signal_simulate(duration=2, frequency=2, sampling_rate=100)
    find_info = nk.signal_findpeaks(sig)
    formatted = nk.signal_formatpeaks(find_info, desired_length=len(sig))
    assert formatted.shape[0] == len(sig)
    assert np.sum(formatted["Peaks"]) == len(find_info["Peaks"])
    assert np.allclose(formatted.loc[find_info["Peaks"], "Height"].to_numpy(), find_info["Height"])

    # Multiple value columns sharing peak_indices / other_indices with NaNs must not corrupt later columns
    info = {
        "SCR_Peaks": np.array([10.0, np.nan, 30.0]),
        "SCR_Amplitude": np.array([1.5, np.nan, 2.5]),
        "SCR_RiseTime": np.array([0.4, np.nan, 0.8]),
        "SCR_Recovery": np.array([np.nan, 20.0, 35.0]),
        "SCR_RecoveryTime": np.array([np.nan, 0.6, 0.9]),
    }
    df = nk.signal_formatpeaks(info, desired_length=40, other_indices=info["SCR_Recovery"])
    assert df.loc[10, "SCR_Amplitude"] == 1.5 and df.loc[30, "SCR_Amplitude"] == 2.5
    assert df.loc[10, "SCR_RiseTime"] == 0.4 and df.loc[30, "SCR_RiseTime"] == 0.8
    assert df.loc[20, "SCR_RecoveryTime"] == 0.6 and df.loc[35, "SCR_RecoveryTime"] == 0.9

    idx_empty, val_empty = _signal_sanitize_indices(None, [1.0, 2.0])
    assert len(idx_empty) == 0 and len(val_empty) == 0
    s_nan = _signal_from_indices(np.array([np.nan, np.nan]), desired_length=10, value=[1.0, 2.0])
    assert np.all(s_nan == 0)
    with pytest.raises(ValueError, match="different from the number of indices"):
        _signal_from_indices([1, 2], desired_length=5, value=[10.0])
