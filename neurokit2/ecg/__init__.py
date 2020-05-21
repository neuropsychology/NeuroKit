"""Submodule for NeuroKit."""

from .ecg_simulate import ecg_simulate
from .ecg_clean import ecg_clean
from .ecg_findpeaks import ecg_findpeaks
from .ecg_peaks import ecg_peaks
from .ecg_segment import ecg_segment
from .ecg_process import ecg_process
from .ecg_plot import ecg_plot
from .ecg_delineate import ecg_delineate
from .ecg_rsp import ecg_rsp
from .ecg_phase import ecg_phase
from .ecg_rsa import ecg_rsa
from .ecg_quality import ecg_quality
from .ecg_eventrelated import ecg_eventrelated
from .ecg_intervalrelated import ecg_intervalrelated
from .ecg_analyze import ecg_analyze

# Aliases
from ..signal import signal_rate as ecg_rate
