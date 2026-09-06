import io
import time
import urllib.parse
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


def read_xdf(
    filename,
    dejitter_timestamps=True,
    synchronize_clocks=True,
    handle_clock_resets=True,
    upsample_factor=2.0,
    fill_method="ffill",
    fill_value=np.nan,
    fill_max=None,
    fillmissing=None,
    interpolation_method="linear",
    timestamp_reset=True,
    timestamp_method="circular",
    mode="precise",
    verbose=True,
    show=None,
    show_start=None,
    show_duration=1.0,
):
    """

    Loads an XDF file, sanitizes stream data, and resamples all streams onto a
    common, synchronized timebase.

    This function handles complex synchronization issues including clock offsets,
    jitter removal (selective or global), and differing sampling rates. It produces
    a single pandas DataFrame containing all aligned data.

    .. note::

        This function requires the *pyxdf* module to be installed. You can install it with
        ``pip install pyxdf``.

    .. warning::

        Note that, as XDF can store streams with different sampling rates and different time stamps,
        **the function will resample all streams to 2 times (default) the highest sampling rate** (to
        minimize aliasing) and then interpolate based on an evenly spaced index. While this is generally safe, it
        may produce unexpected results, particularly if the original stream has large gaps in its time series.
        For more discussion, see `here <https://github.com/xdf-modules/pyxdf/pull/1>`_.

    Parameters
    ----------
    filename : str
        Path to the .xdf file to load.
    dejitter_timestamps : bool or list, optional
        Controls jitter removal (processing of timestamp irregularities).
        If ``bool``, passed directly to pyxdf (``True`` applies to all streams, ``False`` to none).
        If ``list``, a list of stream names (str) or indices (int); dejittering is applied *only*
        to these specific streams. Note: using a list triggers a double-load of the file, increasing
        memory usage and loading time. Default is ``True``.
    synchronize_clocks : bool, optional
        If True, attempts to synchronize clocks using LSL clock offset data.
        Passed to pyxdf.load_xdf. Default is True.
    handle_clock_resets : bool, optional
        If True, handles clock resets (e.g., from hardware restarts) during recording.
        Passed to pyxdf.load_xdf. Default is True.
    upsample_factor : float, optional
        Determines the target sampling rate for the final DataFrame. The target rate
        is calculated as: `max(nominal_srate) * upsample_factor`.
        Higher factors reduce aliasing but increase memory usage. Default is 2.0.
    fill_method : {'ffill', 'bfill', None}, optional
        Method used to fill NaNs arising from resampling (e.g., at the very start of the
        recording before the first sample). Default is ``'ffill'`` (forward fill).
        When ``fill_max`` is set this is automatically overridden to ``None`` so
        that masked gap regions are not silently forward-filled back in. Explicitly pass
        a value to override this behaviour.
    fill_value : float or int, optional
        Value used to fill any NaNs that remain after ``fill_method`` is applied.
        Default is ``np.nan``.
    fill_max : float or None, optional
        If provided, any region of the resampled output that falls inside a gap between
        consecutive original samples larger than this threshold (in seconds) is set back
        to ``NaN`` after interpolation. This prevents the interpolator from silently
        bridging large discontinuities (e.g., paused recordings). When this is set,
        ``fill_method`` is coerced to ``None`` (see above). Default is ``None`` (no
        masking).
    fillmissing : float or None, optional
        Deprecated. Use ``fill_max`` instead.
    interpolation_method : {'linear', 'previous'}, optional
        Method used for interpolating data onto the new timebase.
    timestamp_reset : bool, optional
        If ``True`` (default), shifts all timestamps so the recording starts at t=0.0,
        useful for analysis relative to the start of the specific file.
        If ``False``, preserves the absolute LSL timestamps (Unix epoch), useful when
        synchronizing this data with other files or external clocks.
    timestamp_method : {'circular', 'anchored'}, optional
        Algorithm used to generate the new time axis.
        ``'circular'`` uses a weighted circular mean to find the optimal phase alignment
        across all streams, minimizing global interpolation error.
        ``'anchored'`` aligns the grid strictly to the stream with the highest effective
        sampling rate. Default is ``'circular'``.
    mode : {'precise', 'fast'}, optional
        ``'precise'`` uses float64 for all data, preserving precision but using more memory.
        ``'fast'`` uses float32, reducing memory usage by ~50% but may lose precision
        for very large values. Default is ``'precise'``.
    verbose : bool, optional
        If True, prints progress, target sampling rates, and categorical mappings to console.
        Default is True.
    show : list of str, optional
        A list of channel names to plot for visual quality control after resampling.
        If None, no plots are generated.
    show_start : float, optional
        The start time (in seconds) for the visual control plot window.
        If None, defaults to the middle of the recording.
    show_duration : float, optional
        Duration of the visual control window in seconds. Default is 1 second.

    Returns
    -------
    resampled_df : pandas.DataFrame
        A single DataFrame containing all streams resampled to the common timebase.
        The index is the timestamp (seconds).

    See Also
    --------
    .read_bitalino, .signal_resample

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # data, info = nk.read_xdf("data.xdf")
            # info["stream_names"]               # stream names from the XDF file
            # info["streams"][0]["channel_names"]  # channels for the first stream
      # info["sampling_rates_original"]   # nominal rates per stream
      # info["sampling_rates_effective"]  # computed effective rates per stream
      # info["datetime"]                  # recording datetime from header
    """
    # Handle deprecated fillmissing argument
    if fillmissing is not None:
        warnings.warn(
            "The 'fillmissing' argument is deprecated. Use 'fill_max' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if fill_max is None:
            fill_max = fillmissing

    # When gap masking is active, forward/backward fill would silently erase the
    # masked NaNs.  Coerce fill_method to None so gaps are preserved.
    if fill_max is not None and fill_method in ("ffill", "bfill"):
        warnings.warn(
            f"fill_max is set: fill_method={fill_method!r} would erase masked gap "
            "NaNs by filling through them. Coercing fill_method to None. "
            "Pass fill_method=None explicitly to silence this warning.",
            stacklevel=2,
        )
        fill_method = None

    # Load XDF streams
    streams, header = _load_xdf(
        filename,
        dejitter_timestamps=dejitter_timestamps,
        synchronize_clocks=synchronize_clocks,
        handle_clock_resets=handle_clock_resets,
        verbose=verbose,
    )

    # Sanitize streams
    stream_data = _sanitize_streams(streams, timestamp_reset=timestamp_reset, mode=mode, verbose=verbose)

    if not stream_data:
        raise ValueError("No valid streams remain after sanitization. Check that the file contains streams with timestamps.")

    # Store metadata for the valid streams that contribute to the output.
    info = _build_xdf_info(stream_data, header)

    # Resample and synchronize streams
    resampled_df, target_fs = _synchronize_streams(
        stream_data,
        upsample_factor=upsample_factor,
        fill_method=fill_method,
        fill_value=fill_value,
        max_gap=fill_max,
        interpolation_method=interpolation_method,
        timestamp_method=timestamp_method,
        mode=mode,
        verbose=verbose,
    )
    info["sampling_rate"] = target_fs

    # Quality Control Plots
    if isinstance(show, bool) and show is True:
        show = list(resampled_df.columns)
        if len(show) > 20:
            warnings.warn(f"Plotting all {len(show)} channels. The figure may be very tall.")

    if show is not None and isinstance(show, list) and len(show) > 0:
        _visual_control(
            show,
            stream_data,
            resampled_df,
            window_start=show_start,
            window_duration=show_duration,
            verbose=verbose,
        )
    return resampled_df, info


# =======================================
# Quality Control
# =======================================
def _visual_control(
    show,
    stream_data,
    resampled_df,
    window_start=None,
    window_duration=1.0,
    verbose=True,
):
    # --- Custom Subplot Generation ---
    if verbose:
        print(f"\nGenerating custom plot for {len(show)} specified channels...")

    if window_start is None:
        window_start = resampled_df.index[int(len(resampled_df) / 2)]

    n_plots = len(show)
    # Create a figure with N subplots, sharing the X-axis
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots), sharex=True)

    # Ensure 'axes' is always an iterable array, even if n_plots=1
    if n_plots == 1:
        axes = [axes]

    # Build a lookup map for original stream data (more efficient)
    original_data_map = {}
    for s in stream_data:
        for i, col_name in enumerate(s["columns"]):
            original_data_map[col_name] = {
                "timestamps": s["timestamps"],
                "data": s["data"][:, i],
            }

    # Plot each requested channel on its subplot
    for ax, channel_name in zip(axes, show):
        if channel_name not in original_data_map or channel_name not in resampled_df.columns:
            # Get a sorted list of available columns to help the user
            available_cols = sorted(list(resampled_df.columns))

            warnings.warn(
                f"\n[Visual Control Error] Channel '{channel_name}' not found in data.\n"
                f"Did you mean one of these?\n{available_cols}\n"
            )

            ax.set_title(f"Channel '{channel_name}' - NOT FOUND")
            ax.grid(True)
            continue

        # Get original data and create Series
        original_info = original_data_map[channel_name]
        original_series = pd.Series(
            original_info["data"],
            index=original_info["timestamps"],
            name=channel_name,
        )
        original_series.index.name = "timestamps"

        # Get resampled data (it's already a Series)
        resampled_series = resampled_df[channel_name]

        # Call the visual control helper, passing the specific axis
        _visual_control_channel(
            original_series,
            resampled_series,
            ax=ax,
            window_start=window_start,
            window_duration=window_duration,
        )

    # Tidy up the figure
    fig.tight_layout()
    plt.show()


def _visual_control_channel(original, resampled, window_start=None, window_duration=2.0, ax=None):
    """
    Helper for plotting a window of original vs. resampled data.
    Modified for high-contrast visibility.
    """
    # If no axis is provided, create a new figure and axis
    show_plot = False
    if ax is None:
        plt.figure(figsize=(15, 5))
        ax = plt.gca()
        show_plot = True

    if window_start is None:
        window_start = original.index[int(len(original) / 2)]
    window_end = window_start + window_duration

    # Select the time window
    signal = original[(original.index >= window_start) & (original.index <= window_end)]
    resampled_subset = resampled[(resampled.index >= window_start) & (resampled.index <= window_end)]

    # --- PLOT 1: Resampled Data (The "Fit") ---
    # Bottom layer (zorder=1), Dark Green, thin continuous line
    ax.plot(
        resampled_subset.index,
        resampled_subset,
        "-",  # Continuous line
        color="#D7191C",
        label="resampled",
        alpha=0.7,  # Slightly transparent to see overlaps
        linewidth=1.5,
        zorder=1,  # Draw this first (underneath)
    )

    # --- PLOT 2: Original Data (The "Truth") ---
    # Top layer (zorder=2), Purple, Plus signs
    ax.plot(
        signal.index,
        signal,
        linestyle="--",  # Dashed line (faint) to show connectivity
        linewidth=0.5,  # Very thin connecting line
        marker="+",  # Plus markers
        markersize=7,  # Marker size
        markeredgewidth=1,  # Stroke thickness on the '+'
        color="#2B83BA",
        label="original",
        alpha=1.0,  # Fully opaque
        zorder=2,  # Draw this second (on top)
    )

    ax.legend(loc="upper right")
    ax.set_title(f"Visual Control: {original.name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle=":", alpha=0.6)

    if show_plot:
        plt.show()


# =======================================
# Resampling and Synchronization
# =======================================
def _synchronize_streams(
    stream_data,
    upsample_factor=2.0,
    fill_method: str | None = "ffill",
    fill_value=np.nan,
    max_gap=None,
    interpolation_method="linear",
    timestamp_method="circular",
    mode="precise",
    verbose=True,
):
    """
    - upsample_factor: Factor to multiply max nominal srate by.
    - fill_method: 'ffill', 'bfill', or None
    - fill_value: Value for remaining NaNs
    """
    # --- Compute Target Sampling Rate ---
    target_fs = int(np.max([s["nominal_srate"] for s in stream_data]) * upsample_factor)

    if target_fs == 0:
        raise ValueError(
            "All streams have nominal_srate=0 (irregular/event streams). "
            "Cannot determine a target sampling rate automatically. "
            "Consider adding at least one stream with a non-zero nominal_srate, "
            "or pre-processing the file to assign one."
        )

    if verbose:
        print(f"Target sampling rate: {target_fs} Hz")

    # --- Run Resampling ---
    start_time = time.time()
    resampled_df = _resample_streams(
        stream_data,
        target_fs=target_fs,
        fill_method=fill_method,
        fill_value=fill_value,
        max_gap=max_gap,
        interpolation_method=interpolation_method,
        timestamp_method=timestamp_method,
        mode=mode,
    )
    duration = time.time() - start_time

    if verbose:
        print(f"Resampling complete in {duration:.2f} seconds.")

    return resampled_df, target_fs


def _resample_streams(
    stream_data,
    target_fs,
    fill_method: str | None = "ffill",
    fill_value=np.nan,
    max_gap=None,
    interpolation_method="linear",
    timestamp_method="circular",
    mode="precise",
):
    """
    Resamples and merges multiple XDF streams into a single DataFrame using
    dynamic interpolation (linear or 'previous') and forward-filling.

    Args:
        stream_data (list): List of stream dictionaries from the loading phase.
        target_fs (float): The target sampling rate in Hz.
        fill_method (str): Method for filling NaNs ('ffill', 'bfill', None).
        fill_value (any): Value to fill remaining NaNs (e.g., 0 or np.nan).
        max_gap (float or None): If set, interpolated regions that span
            original-data gaps larger than this value are reset to NaN.

    Returns:
        pd.DataFrame: A single DataFrame with all streams resampled and merged.
    """
    # Unpack column names
    cols = [col for s in stream_data for col in s["columns"]]

    # Create name-to-index mappings for each type
    col_to_idx = {name: i for i, name in enumerate(cols)}

    # Create the target *regular* timestamp grid (once)
    if timestamp_method == "anchored":
        new_ts = _create_timestamps_anchored(stream_data, target_fs)
    elif timestamp_method == "circular":
        new_ts = _create_timestamps_circular(stream_data, target_fs)
    else:
        raise ValueError("timestamp_method must be 'anchored' or 'circular'.")

    # Use mode to determine DataFrame dtype
    target_dtype = np.float32 if mode == "fast" else np.float64

    # Process all streams using the dynamic interpolation function
    data = _interpolate_streams(
        stream_data,
        new_ts,
        cols,
        col_to_idx,
        interpolation_method=interpolation_method,
        max_gap=max_gap,
        dtype=target_dtype,
    )

    # Create DataFrame with specific dtype to save memory
    resampled_df = pd.DataFrame(data, index=new_ts, columns=cols, dtype=target_dtype)

    # Fill NaNs (e.g., at the beginning) and return
    resampled_df = _fill_missing_data(resampled_df, fill_method, fill_value).astype(target_dtype)

    return resampled_df


def _create_timestamps_anchored(stream_data, target_fs):
    """
    Creates a new, regularly spaced timestamp vector "anchored" to the
    stream with the highest effective sampling rate.

    This minimizes interpolation error for the fastest stream by aligning
    the new grid's phase with its existing timestamps. The grid is
    guaranteed to cover the global min/max time of all streams.
    """
    if target_fs <= 0:
        raise ValueError("target_fs must be positive.")

    dt = 1.0 / target_fs

    # 1. Find the global time range (still needed)
    global_min_ts = min([s["timestamps"].min() for s in stream_data])
    global_max_ts = max([s["timestamps"].max() for s in stream_data])

    # 2. Find the "reference" stream (highest effective srate)
    #    We check for len > 1 to avoid divide-by-zero on effective_srate
    #    for single-sample streams.
    try:
        ref_stream = max(
            [s for s in stream_data if len(s["timestamps"]) > 1],
            key=lambda s: s["effective_srate"],
        )
        anchor_ts = ref_stream["timestamps"][0]

    except ValueError:
        # Fallback: No streams have > 1 sample.
        # Revert to the original "ignorant" grid behavior.
        warnings.warn("Could not find a reference stream with > 1 sample. Reverting to un-anchored grid.")
        anchor_ts = global_min_ts

    # 3. Calculate the new start time based on the anchor
    #    We need to find a t_start that is <= global_min_ts
    #    AND is an integer number of steps (dt) away from the anchor.

    # Calculate how far back from the anchor we need to go
    time_before_anchor = anchor_ts - global_min_ts

    # Calculate how many steps (dt) this requires, rounding *up*
    # to ensure we at least cover the global_min_ts.
    # We add a small epsilon to handle potential float precision issues
    # where (time_before_anchor / dt) is *exactly* an integer.
    epsilon = 1e-9
    steps_back = np.ceil((time_before_anchor / dt) + epsilon)

    # Calculate the new, aligned start time
    t_start = anchor_ts - (steps_back * dt)

    # 4. Create the new timestamp vector
    #    The 'stop' condition (global_max_ts + dt) ensures the
    #    last point is >= global_max_ts.
    new_timestamps = np.arange(t_start, global_max_ts + dt, dt)

    return new_timestamps


def _create_timestamps_circular(stream_data, target_fs):
    """
    Creates a new, regularly spaced timestamp vector.

    IMPROVEMENT:
    Instead of snapping strictly to the fastest stream's first sample,
    this uses a 'Weighted Circular Mean' approach. It finds a phase offset
    that minimizes the misalignment across ALL streams, weighted by
    their sampling rates (fast streams pull the grid harder).
    """
    if target_fs <= 0:
        raise ValueError("target_fs must be positive.")

    dt = 1.0 / target_fs

    # 1. Global time boundaries
    valid_streams = [s for s in stream_data if len(s["timestamps"]) > 0]
    if not valid_streams:
        raise ValueError("No valid streams found to generate timestamps.")

    global_min_ts = min([s["timestamps"].min() for s in valid_streams])
    global_max_ts = max([s["timestamps"].max() for s in valid_streams])

    # 2. Calculate Weighted Mean Phase
    # We treat the timestamp's position within a 'dt' cycle as an angle on a circle.
    # We want the average angle.
    sin_sum = 0.0
    cos_sum = 0.0
    total_weight = 0.0

    for s in valid_streams:
        # Use the first timestamp as the phase anchor for this stream
        t0 = s["timestamps"][0]

        # Weight is the effective sampling rate (higher fs = more sensitive to alignment)
        weight = s["effective_srate"]

        # Convert time offset modulo dt to radians [0, 2pi]
        # phase represents how far 'off' this stream is from a generic grid starting at 0
        phase = ((t0 % dt) / dt) * (2 * np.pi)

        sin_sum += weight * np.sin(phase)
        cos_sum += weight * np.cos(phase)
        total_weight += weight

    # 3. Determine Optimal Grid Start
    if total_weight > 0:
        avg_angle = np.arctan2(sin_sum, cos_sum)
        # Convert back to time domain
        # result is in [-pi, pi], map back to [0, dt)
        if avg_angle < 0:
            avg_angle += 2 * np.pi
        optimal_offset = (avg_angle / (2 * np.pi)) * dt
    else:
        optimal_offset = 0

    # 4. Align t_start to this offset, ensuring we start <= global_min_ts
    # We want t_start = k * dt + optimal_offset
    # Find largest k such that t_start <= global_min_ts

    # (global_min_ts - optimal_offset) / dt  gives the number of steps
    steps = np.floor((global_min_ts - optimal_offset) / dt)
    t_start = steps * dt + optimal_offset

    # Safety: ensure we cover the very first sample due to float precision
    if t_start > global_min_ts:
        t_start -= dt

    # 5. Create the vector
    # Add small epsilon to max_ts to ensure inclusion
    new_timestamps = np.arange(t_start, global_max_ts + dt / 2, dt)

    return new_timestamps


def _mask_large_gaps(original_ts, new_ts, interpolated_data, max_gap):
    """
    Set interpolated values back to NaN wherever the new timestamps fall inside
    a gap between consecutive original samples that exceeds *max_gap*.

    Parameters
    ----------
    original_ts : np.ndarray, shape (N,)
        Sorted original timestamps.
    new_ts : np.ndarray, shape (M,)
        Sorted resampled timestamps.
    interpolated_data : np.ndarray, shape (M, C)
        Interpolated data array to modify in-place.
    max_gap : float
        Gaps strictly larger than this threshold are masked.

    Returns
    -------
    interpolated_data : np.ndarray
        The same array with large-gap regions set to NaN.
    """
    gaps = np.diff(original_ts)
    large_gap_mask = gaps > max_gap

    # Fast path: no large gaps at all
    if not np.any(large_gap_mask):
        return interpolated_data

    gap_starts = original_ts[:-1][large_gap_mask]
    gap_ends = original_ts[1:][large_gap_mask]

    # Vectorised: for each new timestamp, find which gap interval it would fall into
    # and check whether that interval is a large gap.
    # np.searchsorted gives the index of the gap whose start is <= new_ts.
    gap_indices = np.searchsorted(gap_starts, new_ts, side="right") - 1
    valid = gap_indices >= 0
    in_large_gap = np.zeros(len(new_ts), dtype=bool)
    in_large_gap[valid] = (new_ts[valid] > gap_starts[gap_indices[valid]]) & (new_ts[valid] < gap_ends[gap_indices[valid]])
    interpolated_data[in_large_gap, :] = np.nan

    return interpolated_data


def _interpolate_streams(
    stream_data,
    new_timestamps,
    all_columns,
    col_to_idx,
    interpolation_method="linear",
    max_gap=None,
    dtype: Any = np.float64,
):
    """
    Performs efficient interpolation.

    Parameters:
    -----------
    mode : str
        "precise" (float64) or "fast" (float32).
    max_gap : float or None
        If not None, regions spanning original-data gaps larger than this
        threshold are reset to NaN after interpolation.
    """

    # 1. Create the empty (NaN-filled) data grid with correct dtype
    resampled_data = np.full((len(new_timestamps), len(all_columns)), np.nan, dtype=dtype)

    # 2. Iterate over each *original* stream and interpolate
    for s in stream_data:
        original_ts = s["timestamps"]
        original_data = s["data"]
        col_indices = [col_to_idx[c] for c in s["columns"]]

        # Handle edge case: stream with 0 or 1 samples
        if len(original_ts) < 2:
            if len(original_ts) == 1:
                # Nearest neighbor "splat" for single points
                insertion_idx = np.searchsorted(new_timestamps, original_ts[0], side="left")
                # Find closest valid index in new grid
                left_idx = np.clip(insertion_idx - 1, 0, len(new_timestamps) - 1)
                right_idx = np.clip(insertion_idx, 0, len(new_timestamps) - 1)

                dist_left = abs(original_ts[0] - new_timestamps[left_idx])
                dist_right = abs(new_timestamps[right_idx] - original_ts[0])

                closest_idx = left_idx if dist_left < dist_right else right_idx
                resampled_data[closest_idx, col_indices] = original_data[0]
            continue

        # --- Determine interpolation kind ---
        # Priority 1: Did sanitization flag this as a categorical/string stream?
        if s.get("force_step_interpolation", False):
            interp_kind = "previous"

        # Priority 2: Does it have very few unique values (e.g., binary triggers)?
        elif np.unique(s["data"]).size <= 2:
            interp_kind = "previous"

        # Priority 3: Standard continuous data
        else:
            interp_kind = interpolation_method

        # --- Interpolation ---
        try:
            n_cols = original_data.shape[1]
            interpolated_data_block = np.empty((len(new_timestamps), n_cols), dtype=dtype)

            if interp_kind == "previous":
                # Zero-order hold (step) interpolation via searchsorted.
                # For each new timestamp, take the value of the most recent
                # original sample that precedes it.
                idx = np.searchsorted(original_ts, new_timestamps, side="right") - 1
                out_of_bounds = (idx < 0) | (new_timestamps > original_ts[-1])
                idx_clipped = np.clip(idx, 0, len(original_ts) - 1)
                interpolated_data_block[:] = original_data[idx_clipped]
                interpolated_data_block[out_of_bounds] = np.nan
            else:
                # Linear interpolation via np.interp (one column at a time).
                # left=nan / right=nan matches interp1d bounds_error=False behaviour.
                for col_i in range(n_cols):
                    interpolated_data_block[:, col_i] = np.interp(
                        new_timestamps,
                        original_ts,
                        original_data[:, col_i],
                        left=np.nan,
                        right=np.nan,
                    )

            # Mask regions that span large gaps in the original data
            if max_gap is not None:
                interpolated_data_block = _mask_large_gaps(
                    original_ts,
                    new_timestamps,
                    interpolated_data_block,
                    max_gap,
                )

            # Place the interpolated data block into the final grid
            resampled_data[:, col_indices] = interpolated_data_block

        except (ValueError, IndexError) as e:
            warnings.warn(f"Interpolation failed for stream '{s['name']}'. Error: {e}")
            continue

    return resampled_data


def _fill_missing_data(resampled_df, fill_method: str | None = "ffill", fill_value=np.nan):
    """
    Fills NaN values in the resampled DataFrame.

    'fill_method':
        - 'ffill': Forward fill
        - 'bfill': Backward fill
        - None: Do not time-based fill
    'fill_value':
        - Value to fill any remaining NaNs (e.g., at the start)
    """
    if fill_method == "ffill":
        resampled_df = resampled_df.ffill()
    elif fill_method == "bfill":
        resampled_df = resampled_df.bfill()

    # Fill any remaining NaNs (e.g., at the very beginning)
    if fill_value is not None and not (isinstance(fill_value, float) and np.isnan(fill_value)):
        resampled_df = resampled_df.fillna(fill_value)

    return resampled_df


# =======================================
# Loading and format sanitization
# =======================================
def _sanitize_streams(streams, timestamp_reset=True, mode="precise", verbose=True):
    """
    Sanitizes XDF streams, handles timestamp offsets, and standardizes data types.

    Parameters:
    -----------
    streams : list
        Raw streams loaded from pyxdf.
    mode : str
        "precise" (default) uses float64 for data.
        "fast" uses float32 to save memory.

    Returns:
    --------
    stream_data : list
        List of processed stream dictionaries.
    """
    # --- Determine Data Type based on Mode ---
    if mode == "fast":
        target_dtype = np.float32
    elif mode == "precise":
        target_dtype = np.float64
    else:
        raise ValueError("mode must be 'precise' or 'fast'")

    # --- Pre-processing & Sanity Checks ---
    # Warn if any stream has no time_stamps
    for i, stream in enumerate(streams):
        name = stream["info"].get("name", ["Unnamed"])[0]
        if len(stream["time_stamps"]) == 0:
            warnings.warn(f"Stream {i} - {name} has no time_stamps. Dropping it.")

    # Drop streams with no timestamps
    streams = [s for s in streams if len(s["time_stamps"]) > 0]

    if not streams:
        warnings.warn("No valid streams found after sanitization.")
        return []

    # If reset is requested, offset is the min_ts. Otherwise, offset is 0.
    timestamp_offset = min([min(s["time_stamps"]) for s in streams]) if timestamp_reset else 0.0

    # Check for duration mismatches
    ts_mins = np.array([stream["time_stamps"].min() for stream in streams])
    ts_maxs = np.array([stream["time_stamps"].max() for stream in streams])
    ts_durations = ts_maxs - ts_mins
    duration_diffs = np.abs(ts_durations[:, np.newaxis] - ts_durations[np.newaxis, :])
    if np.any(duration_diffs > 7200):  # 2 hours
        warnings.warn("Some streams differ in duration by more than 2 hours. This might be indicative of an issue.")

    # --- Convert to common format (list of dicts) ---
    stream_data = []
    for stream in streams:
        # Get column names
        try:
            channels_info = stream["info"]["desc"][0]["channels"][0]["channel"]
            cols = [channels_info[i]["label"][0] for i in range(len(channels_info))]
        except (KeyError, TypeError, IndexError):
            cols = [f"CHANNEL_{i}" for i in range(np.array(stream["time_series"]).shape[1])]
            warnings.warn(f"Using default channel names for stream: {stream['info'].get('name', ['Unnamed'])[0]}")

        name = stream["info"].get("name", ["Unnamed"])[0]
        timestamps = stream["time_stamps"] - timestamp_offset  # Offset applied here
        data = np.array(stream["time_series"])

        # If duplicate timestamps exist, take first occurrence
        unique_ts, unique_indices = np.unique(timestamps, return_index=True)
        data = data[unique_indices]
        timestamps = unique_ts

        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # --- Handle Data Types & Categorical Flags ---
        # We track if a stream was forced to be categorical (string -> int mapped)
        force_step_interpolation = False

        # 1. Attempt direct conversion to target numeric type
        if np.issubdtype(data.dtype, np.number):
            data = data.astype(target_dtype)
        else:
            # Data contains non-numeric objects/strings. Process column by column.
            processed_cols = []

            # Check if we need to force step interpolation for the whole stream
            # (If one channel is categorical, we treat the whole stream group as such to keep alignment)

            for col_idx in range(data.shape[1]):
                column_data = data[:, col_idx]
                try:
                    # Try converting to float (e.g., "1.5" -> 1.5)
                    processed_cols.append(column_data.astype(target_dtype))
                except (ValueError, TypeError):
                    # Conversion failed: This is a string marker channel.
                    force_step_interpolation = True

                    warnings.warn(
                        f"Stream '{name}', column {col_idx} contains non-numeric strings. "
                        f"Converting to integers and forcing 'previous' interpolation."
                    )

                    # Map strings to integers
                    unique_strings = sorted(np.unique(column_data.astype(str)))
                    string_to_int_map = {s: i for i, s in enumerate(unique_strings)}

                    # Print mapping
                    if verbose:
                        col_name = cols[col_idx] if col_idx < len(cols) else f"Idx_{col_idx}"
                        print(f"\n[Categorical Map] Stream: '{name}' | Channel: '{col_name}'")
                        print("-" * 50)
                        for label, val in string_to_int_map.items():
                            print(f"  '{label}' -> {val}")
                        print("-" * 50)

                    mapped_col = np.array([string_to_int_map[s] for s in column_data])
                    processed_cols.append(mapped_col.astype(target_dtype))

            # Recombine columns
            data = np.stack(processed_cols, axis=1)

        if data.shape[0] != len(timestamps):
            warnings.warn(f"Data shape mismatch for stream {name} after unique check. Skipping.")
            continue

        # --- Sanity checks for sampling rates ---
        nominal_srate = float(stream["info"]["nominal_srate"][0])
        effective_srate = len(timestamps) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0

        # Tolerance check
        tol = 0.05 * nominal_srate
        if nominal_srate > 0 and not (nominal_srate - tol <= effective_srate <= nominal_srate + tol):
            warnings.warn(
                f"Stream '{name}': effective sampling rate ({effective_srate:.2f} Hz) "
                f"deviates more than 5% from the nominal rate ({nominal_srate:.2f} Hz). "
                "This may indicate dropped samples or recording issues."
            )

        stream_data.append(
            {
                "timestamps": timestamps,
                "data": data,
                "columns": cols,
                "name": name,
                "type": stream["info"].get("type", [None])[0],
                "nominal_srate": nominal_srate,
                "effective_srate": effective_srate,
                "uid": stream["info"].get("uid", [None])[0],
                "source_id": stream["info"].get("source_id", [None])[0],
                "hostname": stream["info"].get("hostname", [None])[0],
                "force_step_interpolation": force_step_interpolation,  # New Flag
            }
        )

    # --- Handle Duplicate Column Names ---
    all_cols = [col for s in stream_data for col in s["columns"]]
    duplicate_cols = set([col for col in all_cols if all_cols.count(col) > 1])

    if duplicate_cols:
        warnings.warn(f"Duplicate column names found: {duplicate_cols}. Prefixing with stream names.")
        for s in stream_data:
            if any(col in duplicate_cols for col in s["columns"]):
                s["columns"] = [f"{s['name']}_{col}" for col in s["columns"]]

    return stream_data


def _load_xdf(
    filename,
    dejitter_timestamps=True,
    synchronize_clocks=True,
    handle_clock_resets=True,
    verbose=True,
):
    """
    Extended wrapper for pyxdf.load_xdf that allows selective stream dejittering.
    """

    try:
        import pyxdf
    except ImportError as e:
        raise ImportError(
            "The 'pyxdf' module is required for this function to run. Please install it first (`pip install pyxdf`)."
        ) from e

    # Check if filename is a URL string
    if isinstance(filename, str) and urllib.parse.urlparse(filename).scheme in (
        "http",
        "https",
    ):
        if verbose:
            print(f"Downloading XDF from URL: {filename} ...")
        try:
            req = requests.get(filename, stream=True, timeout=10)
            req.raise_for_status()  # Raise error for bad responses (404, 500)
            filename = io.BytesIO(req.content)  # Convert to file-like object
        except requests.exceptions.RequestException as e:
            raise OSError(f"Failed to read XDF file from URL: {filename}") from e

    # Helper to safely rewind if it's a file-like object (BytesIO)
    def _rewind(f):
        if hasattr(f, "seek"):
            f.seek(0)

    # --- Case 1: Boolean (Standard pyxdf behavior) ---
    # If the user passed a simple True/False, we avoid the overhead of double-loading.
    if isinstance(dejitter_timestamps, bool):
        # Good practice to rewind just in case, though technically it's the first read so usually safe.
        _rewind(filename)
        return pyxdf.load_xdf(
            filename,
            synchronize_clocks=synchronize_clocks,
            handle_clock_resets=handle_clock_resets,
            dejitter_timestamps=dejitter_timestamps,
        )

    # --- Case 2: List (Selective Dejittering) ---
    # 1. Load the "Raw" data (Dejitter OFF)
    # We use this as the base object to return.
    _rewind(filename)  # Ensure we start at 0
    streams, header = pyxdf.load_xdf(
        filename,
        synchronize_clocks=synchronize_clocks,
        handle_clock_resets=handle_clock_resets,
        dejitter_timestamps=False,
    )

    # 2. Identify which streams need processing
    # We use a set to store indices to ensure we don't process the same index twice
    # if the user provided both the name and the index for the same stream.
    indices_to_process = set()

    for i, s in enumerate(streams):
        # Extract stream name safely
        stream_name = s["info"].get("name", ["Unnamed"])[0]

        # Check if index is in list OR if name is in list
        if i in dejitter_timestamps or stream_name in dejitter_timestamps:
            indices_to_process.add(i)

    # 3. Optimization Check
    # If no streams matched the user's criteria, return the raw data immediately.
    if not indices_to_process:
        warnings.warn("No matching streams found for dejittering. Make sure you typed the correct name. Returning raw data.")
        return streams, header

    # 4. Load the "Clean" data (Dejitter ON)
    _rewind(filename)  # Reset cursor to zero
    streams_clean, _ = pyxdf.load_xdf(
        filename,
        synchronize_clocks=synchronize_clocks,
        handle_clock_resets=handle_clock_resets,
        dejitter_timestamps=True,
    )

    # 5. Splice the data
    # Replace the raw streams with the clean streams only at the identified indices.
    for i in indices_to_process:
        streams[i] = streams_clean[i]

    return streams, header


def _get_xdf_header_datetime(header):
    """Safely extract the recording datetime from the XDF header."""
    info = header.get("info", {}) if isinstance(header, dict) else {}
    datetime = info.get("datetime") if isinstance(info, dict) else None
    if isinstance(datetime, list):
        return datetime[0] if len(datetime) > 0 else None
    return datetime


def _build_xdf_info(stream_data, header):
    """Build the metadata object returned alongside the resampled DataFrame."""
    streams = []
    for index, stream in enumerate(stream_data):
        timestamps = stream["timestamps"]
        start_time = float(timestamps[0]) if len(timestamps) > 0 else None
        end_time = float(timestamps[-1]) if len(timestamps) > 0 else None
        duration = (
            float(end_time - start_time) if start_time is not None and end_time is not None and len(timestamps) > 1 else 0.0
        )

        streams.append(
            {
                "index": index,
                "name": stream["name"],
                "type": stream.get("type"),
                "channel_count": len(stream["columns"]),
                "channel_names": list(stream["columns"]),
                "nominal_srate": float(stream["nominal_srate"]),
                "effective_srate": float(stream["effective_srate"]),
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "uid": stream.get("uid"),
                "source_id": stream.get("source_id"),
                "hostname": stream.get("hostname"),
            }
        )

    return {
        "sampling_rates_original": [float(stream["nominal_srate"]) for stream in stream_data],
        "sampling_rates_effective": [float(stream["effective_srate"]) for stream in stream_data],
        "datetime": _get_xdf_header_datetime(header),
        "stream_count": len(streams),
        "stream_names": [stream["name"] for stream in streams],
        "stream_types": [stream["type"] for stream in streams],
        "channel_counts": [stream["channel_count"] for stream in streams],
        "channel_names": [stream["channel_names"] for stream in streams],
        "durations": [stream["duration"] for stream in streams],
        "streams": streams,
    }
