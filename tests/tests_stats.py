import numpy as np
import pandas as pd
import pytest

import neurokit2 as nk


# =============================================================================
# Stats
# =============================================================================


def test_standardize():
    rez = np.sum(nk.standardize([1, 1, 5, 2, 1]))
    assert np.allclose(rez, 0, atol=0.0001)

    rez = np.sum(nk.standardize(np.array([1, 1, 5, 2, 1])))
    assert np.allclose(rez, 0, atol=0.0001)

    rez = np.sum(nk.standardize(pd.Series([1, 1, 5, 2, 1])))
    assert np.allclose(rez, 0, atol=0.0001)

    rez = np.sum(nk.standardize([1, 1, 5, 2, 1, 5, 1, 7], robust=True))
    assert np.allclose(rez, 14.8387, atol=0.001)


@pytest.mark.parametrize("kind", ["series", "frame", "mixed_frame"])
@pytest.mark.parametrize("robust", [False, True])
def test_standardize_rolling_pandas_preserves_labels(kind, robust):
    series = pd.Series(
        [1.0, 3.0, 2.0, 5.0],
        index=pd.date_range("2020-01-01", periods=4),
        name="signal",
    )
    scale = 1 / 1.4826 if robust else 1 / np.sqrt(2)
    expected = pd.Series([scale, scale, -scale, scale], index=series.index, name=series.name)
    if kind == "series":
        actual = nk.standardize(series, window=2, robust=robust)
        pd.testing.assert_series_equal(actual, expected)
    else:
        data = series.to_frame()
        if kind == "mixed_frame":
            data["other"] = 2 * series
            data["label"] = ["a", "b", "c", "d"]
        actual = nk.standardize(data, window=2, robust=robust)
        pd.testing.assert_series_equal(actual["signal"], expected)
        if kind == "mixed_frame":
            pd.testing.assert_series_equal(actual["other"], expected.rename("other"))
            pd.testing.assert_series_equal(actual["label"], data["label"])


def test_fit_loess():
    signal = np.cos(np.linspace(start=0, stop=10, num=1000))
    fit, _ = nk.fit_loess(signal, alpha=0.75)
    assert np.allclose(np.mean(signal - fit), -0.0201905899, atol=0.0001)


def test_mad():
    simple_case = [0] * 10
    assert nk.mad(simple_case) == 0

    wikipedia_example = np.array([1, 1, 2, 2, 4, 6, 9])
    constant = 1.42
    assert nk.mad(wikipedia_example, constant=constant) == constant

    negative_wikipedia_example = -wikipedia_example
    assert nk.mad(negative_wikipedia_example, constant=constant) == constant


def create_sample_cluster_data(random_state):
    rng = nk.misc.check_random_state(random_state)

    # generate simple sample data
    K = 5
    points = np.array([[0.0, 0.0], [-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3]])
    centres = np.column_stack((rng.choice(K, size=K, replace=False), rng.choice(K, size=K, replace=False)))
    angles = rng.uniform(0, 2 * np.pi, size=K)
    offset = rng.uniform(size=2)

    # place a cluster at each centre
    data = []
    for i in range(K):
        rotation = np.array([[np.cos(angles[i]), np.sin(angles[i])], [-np.sin(angles[i]), np.cos(angles[i])]])
        data.extend(centres[i] + points @ rotation)
    rng.shuffle(data)

    # shift both data and target centres
    data = np.vstack(data) + offset
    centres = centres + offset

    return data, centres


def test_kmedoids():
    # set random state for reproducible results
    random_state_data = 33
    random_state_clustering = 77

    # create sample data
    data, centres = create_sample_cluster_data(random_state_data)
    K = len(centres)

    # run kmedoids
    res = nk.cluster(data, method="kmedoids", n_clusters=K, random_state=random_state_clustering)

    # check results (sort, then compare rows of res[1] and points)
    assert np.allclose(res[1][np.lexsort(res[1].T)], centres[np.lexsort(centres.T)])


def test_kmeans():
    # set random state for reproducible results
    random_state_data = 54
    random_state_clustering = 76

    # create sample data
    data, centres = create_sample_cluster_data(random_state_data)
    K = len(centres)

    # run kmeans
    res = nk.cluster(data, method="kmeans", n_clusters=K, n_init=1, random_state=random_state_clustering)

    # check results (sort, then compare rows of res[1] and points)
    assert np.allclose(res[1][np.lexsort(res[1].T)], centres[np.lexsort(centres.T)])


def test_cluster_microstates_method_routing():
    rng = np.random.RandomState(42)
    data = rng.normal(size=(20, 4))

    _, _, info = nk.cluster(data, method="ica", n_clusters=2, random_state=42)
    assert info["clustering_function"].func.__name__ == "_cluster_ica"

    _, clusters, info = nk.cluster(data, method="aahc", n_clusters=2, random_state=42)
    assert clusters.shape == (2, data.shape[1])
    assert info["clustering_function"].func.__name__ == "_cluster_aahc"


def test_cor():
    # pearson
    wiki_example_x = np.array([1, 2, 3, 5, 8])
    wiki_example_y = np.array([0.11, 0.12, 0.13, 0.15, 0.18])
    assert nk.cor(wiki_example_x, wiki_example_y) == 1

    # spearman
    wiki_example_x = np.array([106, 100, 86, 101, 99, 103, 97, 113, 112, 110])
    wiki_example_y = np.array([7, 27, 2, 50, 28, 29, 20, 12, 6, 17])
    rez = nk.cor(wiki_example_x, wiki_example_y, "spearman")
    assert np.allclose(rez, -0.175757575, atol=0.0001)

    # kendall
    utd_example_x = np.array([1, 3, 2, 4])
    utd_example_y = np.array([1, 4, 2, 3])
    rez = nk.cor(utd_example_x, utd_example_y, "kendall")
    assert np.allclose(rez, 0.6666666666666669, atol=0.0001)

    # negative test for incorrect 'method' argument
    try:
        rez = nk.cor(wiki_example_x, wiki_example_y, "pearso")
    except ValueError as e:
        print(e)
