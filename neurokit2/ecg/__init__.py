"""Submodule for NeuroKit."""

# Aliases
from ..signal import signal_rate as ecg_rate
from .ecg_analyze import ecg_analyze
from .ecg_clean import ecg_clean
from .ecg_delineate import ecg_delineate
from .ecg_eventrelated import ecg_eventrelated
from .ecg_findpeaks import ecg_findpeaks
from .ecg_intervalrelated import ecg_intervalrelated
from .ecg_peaks import ecg_peaks
from .ecg_phase import ecg_phase
from .ecg_plot import ecg_plot
from .ecg_process import ecg_process
from .ecg_quality import ecg_quality
from .ecg_rsp import ecg_rsp
from .ecg_segment import ecg_segment
from .ecg_simulate import ecg_simulate


__all__ = [
    "ecg_simulate",
    "ecg_clean",
    "ecg_findpeaks",
    "ecg_peaks",
    "ecg_segment",
    "ecg_process",
    "ecg_plot",
    "ecg_delineate",
    "ecg_rsp",
    "ecg_phase",
    "ecg_quality",
    "ecg_eventrelated",
    "ecg_intervalrelated",
    "ecg_analyze",
    "ecg_rate",
]
