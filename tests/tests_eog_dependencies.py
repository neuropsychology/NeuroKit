import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import neurokit2 as nk


@pytest.fixture(scope="module")
def eog_signal():
    return pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "eog_100hz.csv")["vEOG"].to_numpy()


@pytest.mark.parametrize(
    "function, kwargs, origin",
    [
        (nk.eog_findpeaks, {}, "eog_findpeaks"),
        (nk.eog_peaks, {}, "eog_findpeaks"),
        (nk.eog_process, {}, "eog_findpeaks"),
        (nk.eog_clean, {"method": "mne"}, "eog_clean"),
    ],
)
def test_eog_missing_mne(monkeypatch, eog_signal, function, kwargs, origin):
    monkeypatch.setitem(sys.modules, "mne", None)

    with pytest.raises(ImportError, match=rf"NeuroKit error: {origin}\(\).*pip install mne") as error:
        function(eog_signal, sampling_rate=100, **kwargs)

    assert len(error.value.args) == 1
    assert "signal_filter" not in str(error.value)


def test_eog_process_neurokit_without_mne(monkeypatch, eog_signal):
    monkeypatch.setitem(sys.modules, "mne", None)
    try:
        signals, info = nk.eog_process(eog_signal, sampling_rate=100, method="neurokit")
    finally:
        plt.close("all")

    assert len(signals) == len(eog_signal)
    assert len(info["EOG_Blinks"]) == 19
    np.testing.assert_array_equal(np.flatnonzero(signals["EOG_Blinks"]), info["EOG_Blinks"])
