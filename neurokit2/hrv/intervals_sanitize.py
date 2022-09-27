# -*- coding: utf-8 -*-
import numpy as np

from .hrv_utils import _intervals_successive


def intervals_sanitize(intervals, intervals_time=None, remove_missing=True):
    if intervals is None:
        return None, None
    else:
        # Ensure that input is numpy array
        intervals = np.array(intervals)
    if intervals_time is None:
        # Compute the timestamps of the intervals in seconds
        intervals_time = np.nancumsum(intervals / 1000)
    else:
        # Ensure that input is numpy array
        intervals_time = np.array(intervals_time)

        # Confirm that timestamps are in seconds
        successive_intervals = _intervals_successive(intervals, intervals_time=intervals_time)

        if np.all(successive_intervals) is False:
            # If none of the differences between timestamps match
            # the length of the R-R intervals in seconds,
            # try converting milliseconds to seconds
            converted_successive_intervals = _intervals_successive(
                intervals, intervals_time=intervals_time / 1000
            )

            # Check if converting to seconds increased the number of differences
            # between timestamps that match the length of the R-R intervals in seconds
            if len(converted_successive_intervals[converted_successive_intervals]) > len(
                successive_intervals[successive_intervals]
            ):
                # Assume timestamps were passed in milliseconds and convert to seconds
                intervals_time = intervals_time / 1000

    if remove_missing:
        # Remove NaN R-R intervals, if any
        intervals_time = intervals_time[np.isfinite(intervals)]
        intervals = intervals[np.isfinite(intervals)]
    return intervals, intervals_time
