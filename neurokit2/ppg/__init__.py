"""Submodule for NeuroKit."""

# Aliases
from ..signal import signal_rate as ppg_rate
from .ppg_analyze import ppg_analyze
from .ppg_clean import ppg_clean
from .ppg_eventrelated import ppg_eventrelated
from .ppg_findpeaks import ppg_findpeaks
from .ppg_intervalrelated import ppg_intervalrelated
from .ppg_methods import ppg_methods
from .ppg_peaks import ppg_peaks
from .ppg_plot import ppg_plot
from .ppg_process import ppg_process
from .ppg_quality import ppg_quality
from .ppg_segment import ppg_segment
from .ppg_simulate import ppg_simulate

__all__ = [
    "ppg_simulate",
    "ppg_clean",
    "ppg_findpeaks",
    "ppg_peaks",
    "ppg_segment",
    "ppg_rate",
    "ppg_process",
    "ppg_plot",
    "ppg_quality",
    "ppg_methods",
    "ppg_intervalrelated",
    "ppg_eventrelated",
    "ppg_analyze",
]
