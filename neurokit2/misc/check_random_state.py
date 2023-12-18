import copy
import numbers

import numpy as np


def check_random_state(seed=None):
    """**Turn seed into a random number generator**

    Parameters
    ----------
    seed : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator. If seed is None, a numpy.random.Generator is created with fresh,
        unpredictable entropy. If seed is an int, a new numpy.random.RandomState instance is created, seeded with
        seed. If seed is already a Generator or RandomState instance then that instance is used.
        The manin difference between the legacy RandomState class and the new Generator class is that the former
        has better reproducibililty and compatibililty guarantees (it is effectively frozen from NumPy v1.16)
        while the latter has better statistical "randomness" properties and lower computational cost.
        See: https://numpy.org/doc/stable/reference/random/legacy.html for further information.
        Note: to initialise the new Generator class with an integer seed, use, e.g.:
        ``check_random_state(np.random.SeedSequence(123))``.

    Returns
    -------
    rng: numpy.random.Generator or numpy.random.RandomState
        Random number generator.
    """

    # If seed is an integer, use the legacy RandomState class
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    # If seed is already a random number generator class return it as it is
    if isinstance(seed, (np.random.Generator, np.random.RandomState)):
        return seed
    # If seed is something else, use the new Generator class
    return np.random.default_rng(seed)


def spawn_random_state(rng, n_children=1):
    """**Create new independent children random number generators from parent generator/seed**

    Parameters
    ----------
    rng : None, int, numpy.random.RandomState or numpy.random.Generator
        Random number generator to be spawned (numpy.random.RandomState or numpy.random.Generator). If it is None
        or an int seed, then a parent random number generator is first created with ``misc.check_random_state``.
    n_children : int
        Number of children generators to be spawned.

    Returns
    -------
    children_generators : list of generators
        List of children random number generators.

    Examples
    ----------
    * **Example 1**: Simulate data for a cohort of participants

    .. ipython:: python

      import neurokit2 as nk

      master_seed = 42
      n_participants = 8
      participants_RNGs = nk.misc.spawn_random_state(master_seed, n_children=n_participants)
      PPGs = []
      for i in range(n_participants):
          PPGs.append(nk.ppg_simulate(random_state=participants_RNGs[i]))
    """
    rng = check_random_state(rng)

    try:
        # Try to spawn the rng by using the new API
        return rng.spawn(n_children)
    except AttributeError:
        # It looks like this version of numpy does not implement rng.spawn(), so we do its job
        # manually; see: https://github.com/numpy/numpy/pull/23195
        if rng._bit_generator._seed_seq is not None:
            rng_class = type(rng)
            bit_generator_class = type(rng._bit_generator)
            return [
                rng_class(bit_generator_class(seed=s))
                for s in rng._bit_generator._seed_seq.spawn(n_children)
            ]
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


def check_random_state_children(
    random_state_parent, random_state_children, n_children=1
):
    """**Create new independent children random number generators to be used in sub-functions**

    Parameters
    ----------
    random_state_parent : None, int, numpy.random.RandomState or numpy.random.Generator
        Parent's random state (see ``misc.check_random_state``).
    random_state_children : {'legacy', 'spawn'}, None, int, numpy.random.RandomState or numpy.random.Generator
        If ``"legacy"``, use the same random state as the parent (discouraged as it generates dependent random
        streams). If ``"spawn"``, spawn independent children random number generators from the parent random
        state. If any of the other types, generate independent children random number generators from the
        random_state_children provided.
    n_children : int
        Number of children generators to be spawned.

    Returns
    -------
    children_generators : list of generators
        List of children random number generators.
    """
    if random_state_children == "legacy":
        return [copy.copy(random_state_parent) for _ in range(n_children)]
    elif random_state_children == "spawn":
        return spawn_random_state(random_state_parent, n_children)
    else:
        return spawn_random_state(random_state_children, n_children)
