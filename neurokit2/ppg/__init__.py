"""Submodule for NeuroKit."""

# Aliases
from ..signal import signal_rate as ppg_rate
from .ppg_clean import ppg_clean
from .ppg_findpeaks import ppg_findpeaks
from .ppg_plot import ppg_plot
from .ppg_process import ppg_process
from .ppg_simulate import ppg_simulate


__all__ = ["ppg_simulate", "ppg_clean", "ppg_findpeaks", "ppg_rate", "ppg_process", "ppg_plot"]
