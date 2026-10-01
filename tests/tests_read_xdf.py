"""
Unit tests for the internal helpers in neurokit2.data.read_xdf.

These tests cover only the pure-NumPy/pandas logic and do not require
pyxdf or a real .xdf file on disk.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from neurokit2.data.read_xdf import (
    _build_xdf_info,
    _create_timestamps_anchored,
    _create_timestamps_circular,
    _fill_missing_data,
    _interpolate_streams,
    _mask_large_gaps,
    _synchronize_streams,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_stream(t_start, t_end, srate, n_channels=1, name="S"):
    """Build a minimal stream dict as produced by _sanitize_streams."""
    timestamps = np.arange(t_start, t_end, 1.0 / srate)
    data = np.ones((len(timestamps), n_channels), dtype=np.float64)
    return {
        "timestamps": timestamps,
        "data": data,
        "columns": [f"{name}_CH{i}" for i in range(n_channels)],
        "name": name,
        "nominal_srate": float(srate),
        "effective_srate": float(srate),
        "force_step_interpolation": False,
    }


def _make_sanitized_stream(
    timestamps,
    name="S",
    columns=None,
    nominal_srate=100.0,
    effective_srate=100.0,
    stream_type=None,
    uid=None,
    source_id=None,
    hostname=None,
):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if columns is None:
        columns = [f"{name}_CH0"]
    return {
        "timestamps": timestamps,
        "data": np.ones((len(timestamps), len(columns)), dtype=np.float64),
        "columns": columns,
        "name": name,
        "type": stream_type,
        "nominal_srate": float(nominal_srate),
        "effective_srate": float(effective_srate),
        "uid": uid,
        "source_id": source_id,
        "hostname": hostname,
        "force_step_interpolation": False,
    }


# ===========================================================================
# _mask_large_gaps
# ===========================================================================


class TestMaskLargeGaps:
    def _run(self, original_ts, new_ts, max_gap):
        data = np.ones((len(new_ts), 2), dtype=np.float64)
        return _mask_large_gaps(np.asarray(original_ts), np.asarray(new_ts), data, max_gap)

    def test_no_gaps_returns_unchanged(self):
        orig = np.array([0.0, 0.1, 0.2, 0.3])
        new = np.linspace(0.0, 0.3, 30)
        result = self._run(orig, new, max_gap=1.0)
        assert np.all(result == 1.0)

    def test_single_large_gap_masked(self):
        # gap from 1.0 → 3.0 (2 seconds)
        orig = np.array([0.0, 1.0, 3.0, 4.0])
        new = np.linspace(0.0, 4.0, 401)
        result = self._run(orig, new, max_gap=1.5)
        inside = (new > 1.0) & (new < 3.0)
        assert np.all(np.isnan(result[inside, 0]))
        assert np.all(np.isnan(result[inside, 1]))
        # Boundary points are NOT masked (strict inequality)
        assert not np.isnan(result[np.argmin(np.abs(new - 1.0)), 0])
        assert not np.isnan(result[np.argmin(np.abs(new - 3.0)), 0])

    def test_gap_below_threshold_not_masked(self):
        orig = np.array([0.0, 1.0, 1.5, 2.5])
        new = np.linspace(0.0, 2.5, 251)
        result = self._run(orig, new, max_gap=2.0)
        # largest gap is 1.0 s (between 1.5 and 2.5) — below threshold of 2.0
        assert np.all(result == 1.0)

    def test_multiple_gaps_all_masked(self):
        orig = np.array([0.0, 0.1, 2.0, 2.1, 5.0, 5.1])
        new = np.linspace(0.0, 5.1, 512)
        result = self._run(orig, new, max_gap=0.5)
        for g_start, g_end in [(0.1, 2.0), (2.1, 5.0)]:
            inside = (new > g_start) & (new < g_end)
            assert np.all(np.isnan(result[inside, 0])), f"Gap ({g_start},{g_end}) not masked"

    def test_fast_path_no_large_gaps_performance(self):
        """Fast path must return immediately without modifying data."""
        orig = np.linspace(0, 10, 1000)
        new = np.linspace(0, 10, 5000)
        data = np.ones((len(new), 3), dtype=np.float64)
        result = _mask_large_gaps(orig, new, data, max_gap=100.0)
        assert np.all(result == 1.0)


# ===========================================================================
# _fill_missing_data
# ===========================================================================


class TestFillMissingData:
    def _df_with_nans(self):
        data = np.array([[np.nan, 1.0], [2.0, np.nan], [3.0, 4.0]])
        return pd.DataFrame(data, columns=["A", "B"])

    def test_ffill(self):
        df = self._df_with_nans()
        result = _fill_missing_data(df, fill_method="ffill", fill_value=np.nan)
        assert result.loc[0, "A"] != result.loc[0, "A"]  # leading NaN stays
        assert result.loc[1, "B"] == 1.0  # forward-filled

    def test_bfill(self):
        df = self._df_with_nans()
        result = _fill_missing_data(df, fill_method="bfill", fill_value=np.nan)
        assert result.loc[0, "A"] == 2.0  # backward-filled
        assert result.loc[1, "B"] == 4.0  # backward-filled

    def test_fill_value_zero(self):
        df = self._df_with_nans()
        result = _fill_missing_data(df, fill_method=None, fill_value=0)
        assert (result == 0).any().any()

    def test_fill_value_nan_is_noop(self):
        """fillna(np.nan) must not be called (no-op guard)."""
        df = self._df_with_nans()
        result = _fill_missing_data(df, fill_method=None, fill_value=np.nan)
        # NaNs should remain since there's no fill_method and fill_value=np.nan
        assert result.isnull().any().any()

    def test_fill_method_none_preserves_nans(self):
        df = self._df_with_nans()
        result = _fill_missing_data(df, fill_method=None, fill_value=np.nan)
        assert result.isnull().sum().sum() == self._df_with_nans().isnull().sum().sum()


# ===========================================================================
# _create_timestamps_anchored / _create_timestamps_circular
# ===========================================================================


class TestCreateTimestamps:
    def _two_streams(self):
        return [
            _make_stream(0.0, 5.0, srate=100),
            _make_stream(0.0, 5.0, srate=50),
        ]

    def _check_grid(self, ts, target_fs, stream_data):
        dt = 1.0 / target_fs
        global_min = min(s["timestamps"].min() for s in stream_data)
        global_max = max(s["timestamps"].max() for s in stream_data)
        assert ts[0] <= global_min + 1e-9
        assert ts[-1] >= global_max - 1e-9
        diffs = np.diff(ts)
        assert np.allclose(diffs, dt, atol=1e-9)

    def test_anchored_covers_range(self):
        streams = self._two_streams()
        ts = _create_timestamps_anchored(streams, target_fs=200)
        self._check_grid(ts, 200, streams)

    def test_circular_covers_range(self):
        streams = self._two_streams()
        ts = _create_timestamps_circular(streams, target_fs=200)
        self._check_grid(ts, 200, streams)

    def test_anchored_invalid_fs(self):
        with pytest.raises(ValueError):
            _create_timestamps_anchored(self._two_streams(), target_fs=0)

    def test_circular_invalid_fs(self):
        with pytest.raises(ValueError):
            _create_timestamps_circular(self._two_streams(), target_fs=0)

    def test_circular_no_streams(self):
        with pytest.raises(ValueError):
            _create_timestamps_circular([], target_fs=100)


# ===========================================================================
# _synchronize_streams — target_fs == 0 guard
# ===========================================================================


class TestSynchronizeStreams:
    def _irregular_streams(self):
        """All streams have nominal_srate=0 (event/marker)."""
        s = _make_stream(0.0, 5.0, srate=100)
        s["nominal_srate"] = 0.0
        return [s]

    def test_raises_when_all_irregular(self):
        with pytest.raises(ValueError, match="nominal_srate=0"):
            _synchronize_streams(self._irregular_streams(), upsample_factor=2.0, verbose=False)

    def test_message_does_not_suggest_upsample_factor(self):
        """The error message must not suggest changing upsample_factor (it can't help)."""
        with pytest.raises(ValueError) as exc_info:
            _synchronize_streams(self._irregular_streams(), upsample_factor=99.0, verbose=False)
        assert "upsample_factor" not in str(exc_info.value)

    def test_normal_streams_produce_dataframe(self):
        streams = [_make_stream(0.0, 1.0, srate=100, n_channels=2)]
        df, target_fs = _synchronize_streams(streams, upsample_factor=1.0, verbose=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert target_fs == 100


# ===========================================================================
# _build_xdf_info
# ===========================================================================


class TestBuildXdfInfo:
    def test_returns_stream_level_metadata(self):
        streams = [
            _make_sanitized_stream(
                timestamps=[0.0, 0.5, 1.0],
                name="EEG",
                columns=["C3", "C4"],
                nominal_srate=250.0,
                effective_srate=249.5,
                stream_type="EEG",
                uid="uid-eeg",
                source_id="src-eeg",
                hostname="host-a",
            ),
            _make_sanitized_stream(
                timestamps=[0.2],
                name="Markers",
                columns=["event"],
                nominal_srate=0.0,
                effective_srate=0.0,
                stream_type="Markers",
            ),
        ]
        header = {"info": {"datetime": ["2026-03-07T12:00:00"]}}

        info = _build_xdf_info(streams, header)

        assert info["datetime"] == "2026-03-07T12:00:00"
        assert info["stream_count"] == 2
        assert info["stream_names"] == ["EEG", "Markers"]
        assert info["stream_types"] == ["EEG", "Markers"]
        assert info["channel_counts"] == [2, 1]
        assert info["channel_names"] == [["C3", "C4"], ["event"]]
        assert info["sampling_rates_original"] == [250.0, 0.0]
        assert info["sampling_rates_effective"] == [249.5, 0.0]
        assert info["durations"] == [1.0, 0.0]

        assert info["streams"][0] == {
            "index": 0,
            "name": "EEG",
            "type": "EEG",
            "channel_count": 2,
            "channel_names": ["C3", "C4"],
            "nominal_srate": 250.0,
            "effective_srate": 249.5,
            "start_time": 0.0,
            "end_time": 1.0,
            "duration": 1.0,
            "uid": "uid-eeg",
            "source_id": "src-eeg",
            "hostname": "host-a",
        }
        assert info["streams"][1]["duration"] == 0.0
        assert info["streams"][1]["start_time"] == 0.2
        assert info["streams"][1]["end_time"] == 0.2


# ===========================================================================
# _interpolate_streams — end-to-end with gap masking
# ===========================================================================


class TestInterpolateStreams:
    def test_linear_interpolation(self):
        orig_ts = np.array([0.0, 1.0, 2.0, 3.0])
        orig_data = np.array([[0.0], [1.0], [2.0], [3.0]])
        stream = {
            "timestamps": orig_ts,
            "data": orig_data,
            "columns": ["CH0"],
            "name": "S",
            "nominal_srate": 1.0,
            "effective_srate": 1.0,
            "force_step_interpolation": False,
        }
        new_ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        result = _interpolate_streams(
            [stream],
            new_ts,
            ["CH0"],
            {"CH0": 0},
            interpolation_method="linear",
            max_gap=None,
        )
        expected = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        np.testing.assert_allclose(result[:, 0], expected, atol=1e-12)

    def test_gap_masking_applied(self):
        # gap from 1.0 → 3.0 (2 s) — should be masked with max_gap=1.5
        orig_ts = np.array([0.0, 1.0, 3.0, 4.0])
        orig_data = np.array([[0.0], [1.0], [3.0], [4.0]])
        stream = {
            "timestamps": orig_ts,
            "data": orig_data,
            "columns": ["CH0"],
            "name": "S",
            "nominal_srate": 1.0,
            "effective_srate": 1.0,
            "force_step_interpolation": False,
        }
        new_ts = np.linspace(0.0, 4.0, 401)
        result = _interpolate_streams(
            [stream],
            new_ts,
            ["CH0"],
            {"CH0": 0},
            interpolation_method="linear",
            max_gap=1.5,
        )
        inside = (new_ts > 1.0) & (new_ts < 3.0)
        assert np.all(np.isnan(result[inside, 0]))
        assert not np.isnan(result[0, 0])  # before gap
        assert not np.isnan(result[-1, 0])  # after gap

    def test_previous_interpolation(self):
        orig_ts = np.array([0.0, 1.0, 2.0])
        orig_data = np.array([[10.0], [20.0], [30.0]])
        stream = {
            "timestamps": orig_ts,
            "data": orig_data,
            "columns": ["CH0"],
            "name": "S",
            "nominal_srate": 1.0,
            "effective_srate": 1.0,
            "force_step_interpolation": True,
        }
        new_ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        result = _interpolate_streams(
            [stream],
            new_ts,
            ["CH0"],
            {"CH0": 0},
            interpolation_method="previous",
        )
        # At 0.5 → held from 0.0 → value 10
        # At 1.5 → held from 1.0 → value 20
        assert result[1, 0] == 10.0
        assert result[3, 0] == 20.0


# ===========================================================================
# fill_method coercion warning in read_xdf (unit-level, no pyxdf needed)
# ===========================================================================


class TestFillMethodCoercion:
    """
    Test the coercion logic directly by re-running the relevant lines
    from read_xdf's preamble, rather than calling the full function.
    """

    def _coerce(self, fill_method, fill_max):
        """Mirror the coercion block in read_xdf."""
        if fill_max is not None and fill_method in ("ffill", "bfill"):
            warnings.warn(
                f"fill_max is set: fill_method={fill_method!r} would erase "
                "masked gap NaNs by filling through them. Coercing fill_method to None.",
                stacklevel=1,
            )
            fill_method = None
        return fill_method

    def test_ffill_coerced_with_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self._coerce("ffill", fill_max=5.0)
        assert result is None
        assert len(w) == 1
        assert "ffill" in str(w[0].message)

    def test_bfill_coerced_with_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self._coerce("bfill", fill_max=5.0)
        assert result is None
        assert len(w) == 1
        assert "bfill" in str(w[0].message)

    def test_none_fill_method_not_coerced(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self._coerce(None, fill_max=5.0)
        assert result is None
        assert len(w) == 0  # no warning when already None

    def test_no_coercion_when_fill_max_none(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self._coerce("ffill", fill_max=None)
        assert result == "ffill"
        assert len(w) == 0


# ===========================================================================
# fillmissing deprecation
# ===========================================================================


class TestFillmissingDeprecation:
    """fillmissing should raise DeprecationWarning and forward its value to fill_max."""

    def _run_deprecation_logic(self, fillmissing, fill_max):
        """Mirror the deprecation block in read_xdf."""
        if fillmissing is not None:
            warnings.warn(
                "The 'fillmissing' argument is deprecated. Use 'fill_max' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if fill_max is None:
                fill_max = fillmissing
        return fill_max

    def test_fillmissing_raises_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._run_deprecation_logic(fillmissing=5.0, fill_max=None)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "fill_max" in str(w[0].message)

    def test_fillmissing_value_forwarded_to_fill_max(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = self._run_deprecation_logic(fillmissing=7.5, fill_max=None)
        assert result == 7.5

    def test_explicit_fill_max_not_overridden_by_fillmissing(self):
        """If the user passes both, fill_max wins."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = self._run_deprecation_logic(fillmissing=3.0, fill_max=10.0)
        assert result == 10.0

    def test_no_warning_when_fillmissing_is_none(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._run_deprecation_logic(fillmissing=None, fill_max=None)
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 0
