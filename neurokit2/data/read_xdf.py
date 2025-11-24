import warnings
import numpy as np
import time
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import scipy.interpolate
import urllib.parse
import requests
import io


def read_xdf(
    filename,
    dejitter_timestamps=True,
    synchronize_clocks=True,
    handle_clock_resets=True,
    upsample_factor=2.0,
    fill_method="ffill",
    fill_value=0,
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
        - If bool: Passed directly to pyxdf (True applies to all streams, False to none).
        - If list: A list of stream names (str) or indices (int). Dejittering is
          applied *only* to these specific streams.
          Note: Using a list triggers a double-load of the file, increasing memory
          usage and loading time. Default is True.
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
        Method used to fill NaNs arising from resampling (e.g., zero-order hold).
        Default is 'ffill' (forward fill).
    fill_value : float or int, optional
        Value used to fill remaining NaNs (e.g., at the start of the recording before
        the first sample). Default is 0.
    fillmissing : float or int, optional
        DEPRECATED: This argument is deprecated and has no direct equivalent in the new
        implementation. It previously controlled filling of gaps larger than a threshold.
    interpolation_method : {'linear', 'previous'}, optional
        Method used for interpolating data onto the new timebase.
    timestamp_reset : bool, optional
        - If True (default): Shifts all timestamps so the recording starts at t=0.0.
          Useful for analysis relative to the start of the specific file.
        - If False: Preserves the absolute LSL timestamps (Unix epoch). Useful when
          synchronizing this data with other files or external clocks.
    timestamp_method : {'circular', 'anchored'}, optional
        Algorithm used to generate the new time axis.
        - 'circular': Uses a weighted circular mean to find the optimal phase alignment
          across all streams. Minimizes global interpolation error.
        - 'anchored': Aligns the grid strictly to the stream with the highest effective
          sampling rate.
        Default is 'circular'.
    mode : {'precise', 'fast'}, optional
        - 'precise': Uses float64 for all data. Preserves precision but uses more memory.
        - 'fast': Uses float32. Reduces memory usage by ~50% but may lose precision
          for very large values.
        Default is 'precise'.
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
      # sampling_rate = info["sampling_rate"]
    """
    try:
        import pyxdf
    except ImportError as e:
        raise ImportError(
            "The 'pyxdf' module is required for this function to run. ",
            "Please install it first (`pip install pyxdf`).",
        ) from e

    # DEPRECATION WARNING
    if fillmissing is not None:
        warnings.warn(
            "The 'fillmissing' argument is deprecated and has no direct equivalent in the new optimized implementation. "
            "This function uses 'scipy.interpolate' which interpolates across all gaps regardless of duration. "
            "If you need to mask large gaps, please do so on the returned DataFrame.",
            Category=DeprecationWarning,
            stacklevel=2,
        )

    # Load XDF streams
    streams, header = _load_xdf(
        filename,
        dejitter_timestamps=dejitter_timestamps,
        synchronize_clocks=synchronize_clocks,
        handle_clock_resets=handle_clock_resets,
        verbose=verbose,
    )

    # Store metadata
    info = {
        "sampling_rates_original": [
            float(s["info"]["nominal_srate"][0]) for s in streams
        ],
        "sampling_rates_effective": [
            float(s["info"]["effective_srate"]) for s in streams
        ],
        "datetime": header["info"]["datetime"][0],
    }

    # Sanitize streams
    stream_data = _sanitize_streams(
        streams, timestamp_reset=timestamp_reset, mode=mode, verbose=verbose
    )

    # Resample and synchronize streams
    resampled_df = _synchronize_streams(
        stream_data,
        upsample_factor=upsample_factor,
        fill_method=fill_method,
        fill_value=fill_value,
        interpolation_method=interpolation_method,
        timestamp_method=timestamp_method,
        mode=mode,
    )

    # Quality Control Plots
    if isinstance(show, bool) and show is True:
        show = list(resampled_df.columns)
        if len(show) > 20:
            warnings.warn(
                f"Plotting all {len(show)} channels. The figure may be very tall."
            )

    if show is not None and isinstance(show, list) and len(show) > 0:
        _visual_control(
            show,
            stream_data,
            resampled_df,
            window_start=show_start,
            window_duration=show_duration,
        )
    return resampled_df, info


# =======================================
# Quality Control
# =======================================
def _visual_control(
    show, stream_data, resampled_df, window_start=None, window_duration=1.0
):
    # --- Custom Subplot Generation ---
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
        if (
            channel_name not in original_data_map
            or channel_name not in resampled_df.columns
        ):
            # --- FIX START: Enhanced Debug Message ---
            # Get a sorted list of available columns to help the user
            available_cols = sorted(list(resampled_df.columns))

            warnings.warn(
                f"\n[Visual Control Error] Channel '{channel_name}' not found in data.\n"
                f"Did you mean one of these?\n{available_cols}\n"
            )
            # --- FIX END ---

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


def _visual_control_channel(
    original, resampled, window_start=None, window_duration=2.0, ax=None
):
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
    resampled_subset = resampled[
        (resampled.index >= window_start) & (resampled.index <= window_end)
    ]

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
    fill_method="ffill",
    fill_value=0,
    interpolation_method="linear",
    timestamp_method="circular",
    mode="precise",
):
    """
    - upsample_factor: Factor to multiply max nominal srate by.
    - fill_method: 'ffill', 'bfill', or None
    - fill_value: Value for remaining NaNs
    - show (list or None): List of channel names to plot on a single figure.
                           If None, no plots are generated.
    """
    # --- Compute Target Sampling Rate ---
    target_fs = int(np.max([s["nominal_srate"] for s in stream_data]) * upsample_factor)

    print(f"Target sampling rate: {target_fs} Hz")

    # --- Run Resampling ---
    start_time = time.time()
    resampled_df = _resample_streams(
        stream_data,
        target_fs=target_fs,
        fill_method=fill_method,
        fill_value=fill_value,
        interpolation_method=interpolation_method,
        timestamp_method=timestamp_method,
        mode=mode,
    )
    duration = time.time() - start_time

    print(f"Resampling complete in {duration:.2f} seconds.")

    return resampled_df


def _resample_streams(
    stream_data,
    target_fs,
    fill_method="ffill",
    fill_value=0,
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
        dtype=target_dtype,
    )

    # Create DataFrame with specific dtype to save memory
    resampled_df = pd.DataFrame(data, index=new_ts, columns=cols, dtype=target_dtype)

    # Fill NaNs (e.g., at the beginning) and return
    resampled_df = _fill_missing_data(resampled_df, fill_method, fill_value).astype(
        target_dtype
    )

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
        warnings.warn(
            "Could not find a reference stream with > 1 sample. "
            "Reverting to un-anchored grid."
        )
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


def _interpolate_streams(
    stream_data,
    new_timestamps,
    all_columns,
    col_to_idx,
    interpolation_method="linear",
    dtype=np.float64,
):
    """
    Performs efficient interpolation.

    Parameters:
    -----------
    mode : str
        "precise" (float64) or "fast" (float32).
    """

    # 1. Create the empty (NaN-filled) data grid with correct dtype
    resampled_data = np.full(
        (len(new_timestamps), len(all_columns)), np.nan, dtype=dtype
    )

    # 2. Iterate over each *original* stream and interpolate
    for s in stream_data:
        original_ts = s["timestamps"]
        original_data = s["data"]
        col_indices = [col_to_idx[c] for c in s["columns"]]

        # Handle edge case: stream with 0 or 1 samples
        if len(original_ts) < 2:
            if len(original_ts) == 1:
                # Nearest neighbor "splat" for single points
                insertion_idx = np.searchsorted(
                    new_timestamps, original_ts[0], side="left"
                )
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
            # assume_sorted=True improves performance significantly
            interpolator = scipy.interpolate.interp1d(
                original_ts,
                original_data,
                axis=0,
                kind=interp_kind,
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )

            # Apply the interpolator to the new timestamps
            interpolated_data_block = interpolator(new_timestamps)

            # Ensure the block matches the target dtype
            if interpolated_data_block.dtype != dtype:
                interpolated_data_block = interpolated_data_block.astype(dtype)

            # Place the interpolated data block into the final grid
            resampled_data[:, col_indices] = interpolated_data_block

        except ValueError as e:
            warnings.warn(f"Interpolation failed for stream '{s['name']}'. Error: {e}")
            continue

    return resampled_data


def _fill_missing_data(resampled_df, fill_method="ffill", fill_value=0):
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
    if fill_value is not None:
        resampled_df = resampled_df.fillna(fill_value)

    # After filling, infer the best possible dtypes to silence FutureWarning
    # copy=False modifies the df in place if possible
    resampled_df = resampled_df.infer_objects(copy=False)

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
    timestamp_offset = (
        min([min(s["time_stamps"]) for s in streams]) if timestamp_reset else 0.0
    )

    # Check for duration mismatches
    ts_mins = np.array([stream["time_stamps"].min() for stream in streams])
    ts_maxs = np.array([stream["time_stamps"].max() for stream in streams])
    ts_durations = ts_maxs - ts_mins
    duration_diffs = np.abs(ts_durations[:, np.newaxis] - ts_durations[np.newaxis, :])
    if np.any(duration_diffs > 7200):  # 2 hours
        warnings.warn(
            "Some streams differ in duration by more than 2 hours. This might be indicative of an issue."
        )

    # --- Convert to common format (list of dicts) ---
    stream_data = []
    for stream in streams:
        # Get column names
        try:
            channels_info = stream["info"]["desc"][0]["channels"][0]["channel"]
            cols = [channels_info[i]["label"][0] for i in range(len(channels_info))]
        except (KeyError, TypeError, IndexError):
            cols = [
                f"CHANNEL_{i}" for i in range(np.array(stream["time_series"]).shape[1])
            ]
            warnings.warn(
                f"Using default channel names for stream: {stream['info'].get('name', ['Unnamed'])[0]}"
            )

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
                        col_name = (
                            cols[col_idx] if col_idx < len(cols) else f"Idx_{col_idx}"
                        )
                        print(
                            f"\n[Categorical Map] Stream: '{name}' | Channel: '{col_name}'"
                        )
                        print("-" * 50)
                        for label, val in string_to_int_map.items():
                            print(f"  '{label}' -> {val}")
                        print("-" * 50)

                    mapped_col = np.array([string_to_int_map[s] for s in column_data])
                    processed_cols.append(mapped_col.astype(target_dtype))

            # Recombine columns
            data = np.stack(processed_cols, axis=1)

        if data.shape[0] != len(timestamps):
            warnings.warn(
                f"Data shape mismatch for stream {name} after unique check. Skipping."
            )
            continue

        # --- Sanity checks for sampling rates ---
        nominal_srate = float(stream["info"]["nominal_srate"][0])
        effective_srate = (
            len(timestamps) / (timestamps[-1] - timestamps[0])
            if len(timestamps) > 1
            else 0
        )

        # Tolerance check
        tol = 0.05 * nominal_srate
        if nominal_srate > 0 and not (
            nominal_srate - tol <= effective_srate <= nominal_srate + tol
        ):
            # Just a warning, not an error
            pass

        stream_data.append(
            {
                "timestamps": timestamps,
                "data": data,
                "columns": cols,
                "name": name,
                "nominal_srate": nominal_srate,
                "effective_srate": effective_srate,
                "force_step_interpolation": force_step_interpolation,  # New Flag
            }
        )

    # --- Handle Duplicate Column Names ---
    all_cols = [col for s in stream_data for col in s["columns"]]
    duplicate_cols = set([col for col in all_cols if all_cols.count(col) > 1])

    if duplicate_cols:
        warnings.warn(
            f"Duplicate column names found: {duplicate_cols}. Prefixing with stream names."
        )
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
            raise IOError(f"Failed to read XDF file from URL: {filename}") from e

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
        warnings.warn(
            "No matching streams found for dejittering. Make sure you typed the correct name. Returning raw data."
        )
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
