"""Submodule for NeuroKit."""

from .eog_clean import eog_clean
from .eog_peaks import eog_peaks
from .eog_process import eog_process


__all__ = ["eog_clean", "eog_peaks", "eog_process"]
