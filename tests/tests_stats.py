import numpy as np
import pandas as pd

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
