import numpy as np
import pandas as pd

import neurokit2 as nk
from neurokit2.microstates.microstates_segment import _microstates_segment_runsegmentation
from neurokit2.stats.cluster_quality import _cluster_quality_gev


# =============================================================================
# Peaks
# =============================================================================


def test_microstates_peaks():
    # Load eeg data and calculate gfp
    eeg = nk.mne_data("filt-0-40_raw")
    gfp = nk.eeg_gfp(eeg)

    # Find peaks
    peaks_nk = nk.microstates_peaks(eeg, distance_between=0.01)

    # Test with alternative method taken from Frederic
    # https://github.com/Frederic-vW/eeg_microstates/blob/master/eeg_microstates.py
    def locmax(x):
        dx = np.diff(x)  # discrete 1st derivative
        zc = np.diff(np.sign(dx))  # zero-crossings of dx
        m = 1 + np.where(zc == -2)[0]  # indices of local max.
        return m

    peaks_frederic = locmax(gfp)

    assert all(elem in peaks_frederic for elem in peaks_nk)  # only works when distance_between = 0.01


def test_microstates_peaks_training_selection():
    eeg = np.random.RandomState(42).normal(size=(4, 20))

    np.testing.assert_array_equal(nk.microstates_peaks(eeg, gfp="all"), np.arange(20))
    np.testing.assert_array_equal(nk.microstates_peaks(eeg, gfp=5), [0, 4, 8, 12, 16])

    gfp = np.zeros(20)
    gfp[[7, 15]] = 1
    np.testing.assert_array_equal(nk.microstates_peaks(eeg, gfp=gfp, sampling_rate=100), [7, 15])


def test_microstates_clean_input_shapes_and_standardization():
    eeg = np.arange(60, dtype=float).reshape(3, 20) + np.arange(3)[:, None] ** 2
    data, indices, _, _ = nk.microstates_clean(pd.DataFrame(eeg), train="all", standardize_eeg=True)

    assert data.shape == eeg.shape
    np.testing.assert_array_equal(indices, np.arange(eeg.shape[1]))
    np.testing.assert_allclose(np.mean(data, axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(np.std(data, axis=1, ddof=1), 1, atol=1e-12)


def test_microstates_clean_mne_epochs():
    import mne

    rng = np.random.RandomState(42)
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq=100, ch_types="eeg")
    epochs = mne.EpochsArray(rng.normal(size=(2, 4, 20)), info, verbose=False)

    data, indices, gfp, returned_info = nk.microstates_clean(epochs, train="all", standardize_eeg=False)

    assert data.shape == (4, 40)
    np.testing.assert_array_equal(indices, np.arange(40))
    assert gfp.shape == (40,)
    assert returned_info is epochs.info

    output = nk.microstates_segment(epochs, n_microstates=2, train="all", n_runs=2, random_state=42)
    assert output["Sequence"].shape == (40,)


def test_microstates_segment_keeps_clustering_kwargs_out_of_preprocessing():
    eeg = np.random.RandomState(42).normal(size=(4, 30))
    output = nk.microstates_segment(
        eeg,
        n_microstates=2,
        train="all",
        method="kmeans",
        n_init=2,
        random_state=42,
    )

    assert output["Sequence"].shape == (eeg.shape[1],)
    assert output["Info_algorithm"]["sklearn_model"].n_init == 2


def test_microstates_classify_preserves_map_sequence_correspondence():
    microstates = np.array(
        [
            [-0.403, 1.222, 0.208, 0.977, 0.356],
            [0.707, 0.011, 1.786, 0.127, 0.402],
            [1.883, -1.348, -1.270, 0.969, -1.173],
        ]
    )
    segmentation = np.array([0, 1, 2])

    classified, reordered, order = nk.microstates_classify(segmentation, microstates, return_order=True)

    np.testing.assert_array_equal(order, [1, 2, 0])
    np.testing.assert_array_equal(classified, [2, 0, 1])
    for old_label, new_label in zip(segmentation, classified):
        np.testing.assert_allclose(microstates[old_label], reordered[new_label])


def test_microstates_segmentation_cv_gev_order_and_map_scale():
    rng = np.random.RandomState(42)
    eeg = rng.normal(size=(5, 50))
    output = nk.microstates_segment(
        eeg,
        n_microstates=3,
        train="all",
        criterion="cv",
        n_runs=2,
        random_state=42,
    )

    assert isinstance(output["Info_algorithm"], dict)
    _, expected_gev = _cluster_quality_gev(
        eeg.T,
        output["Microstates"],
        output["Sequence"],
        sd=output["GFP"],
        n_clusters=3,
    )
    np.testing.assert_allclose(output["GEV_per_microstate"], expected_gev)

    maps = rng.normal(size=(3, 5))
    scaled_maps = maps * np.array([1, 10, 100])[:, None]
    sequence, _, _, _ = _microstates_segment_runsegmentation(eeg, maps, output["GFP"], n_microstates=3)
    scaled_sequence, _, _, _ = _microstates_segment_runsegmentation(
        eeg, scaled_maps, output["GFP"], n_microstates=3
    )
    np.testing.assert_array_equal(sequence, scaled_sequence)


def test_microstates_segment_documented_clustering_methods():
    eeg = np.random.RandomState(42).normal(size=(6, 40))

    for method in ["pca", "ica", "aahc"]:
        output = nk.microstates_segment(
            eeg,
            n_microstates=2,
            train="all",
            method=method,
            random_state=42,
        )
        assert output["Microstates"].shape == (2, eeg.shape[0])
        assert output["Sequence"].shape == (eeg.shape[1],)
        assert np.isfinite(output["GEV"])
