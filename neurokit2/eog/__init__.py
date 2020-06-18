"""Submodule for NeuroKit."""

from .eog_clean import eog_clean
from .eog_features import eog_features
from .eog_findpeaks import eog_findpeaks
from .eog_process import eog_process


__all__ = ["eog_clean", "eog_features", "eog_findpeaks", "eog_process"]
