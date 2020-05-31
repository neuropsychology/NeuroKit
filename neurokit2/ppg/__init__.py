"""Submodule for NeuroKit."""

from .ppg_simulate import ppg_simulate
from .ppg_clean import ppg_clean
from .ppg_findpeaks import ppg_findpeaks

# Aliases
from ..signal import signal_rate as ppg_rate

__all__ = ["ppg_simulate", "ppg_clean", "ppg_findpeaks", "ppg_rate"]
