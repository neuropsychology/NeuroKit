"""Submodule for NeuroKit."""

from .eog_clean import eog_clean
from .eog_extract import eog_extract
from .eog_process import eog_process


__all__ = ["eog_clean", "eog_extract", "eog_process"]
