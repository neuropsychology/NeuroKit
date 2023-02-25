import copy
import numbers

import numpy as np


def check_rng(seed=None):
    # If seed is an integer, use the legacy RandomState generator, which has better compatibililty
    # guarantees but worse statistical "randomness" properties and higher computational cost
    # See: https://numpy.org/doc/stable/reference/random/legacy.html
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    # If seed is already a random number generator return it as it is
    if isinstance(seed, (np.random.Generator, np.random.RandomState)):
        return seed
    # If seed is something else, use the new Generator class
    # Note: to initialise the new generator class with an integer seed, use, e.g.:
    # check_rng(np.random.SeedSequence(123))
    return np.random.default_rng(seed)


def spawn_rng(rng, n_children=1):
    rng = check_rng(rng)

    try:
        # Try to spawn the rng by using the new API
        return rng.spawn(n_children)
    except AttributeError:
        # It looks like this version of numpy does not implement rng.spawn(), so we do its job
        # manually; see: https://github.com/numpy/numpy/pull/23195
        if rng._bit_generator._seed_seq is not None:
            rng_class = type(rng)
            bit_generator_class = type(rng._bit_generator)
            return [rng_class(bit_generator_class(seed=s)) for s in rng._bit_generator._seed_seq.spawn(n_children)]
    except TypeError:
        # The rng does not support spawning through SeedSequence, see below
        pass

    # Implement a rudimentary but reproducible substitute for spawning rng's that also works for
    # RandomState with the legacy MT19937 bit generator
    # NOTE: Spawning the same generator multiple times is not supported (may lead to mutually
    # dependent spawned generators). Spawning the children (in a tree structure) is allowed.

    # Start by creating an rng to sample integers (to be used as seeds for the children) without
    # advancing the original rng
    temp_rng = rng._bit_generator.jumped()
    # Generate and return children initialised with the seeds obtained from temp_rng
    return [np.random.RandomState(seed=s) for s in temp_rng.random_raw(n_children)]


def get_children_rng(parent_random_state, children_random_state, n_children=1):
    if children_random_state == "legacy":
        return [copy.copy(parent_random_state) for _ in range(n_children)]
    elif children_random_state == "spawn":
        return spawn_rng(parent_random_state, n_children)
    else:
        return spawn_rng(children_random_state, n_children)
