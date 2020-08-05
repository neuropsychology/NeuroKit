"""Submodule for NeuroKit."""
import functools

from .complexity_delay import complexity_delay
from .complexity_dimension import complexity_dimension

# Utils
from .complexity_embedding import complexity_embedding
from .complexity_optimize import complexity_optimize
from .complexity_r import complexity_r
from .complexity_simulate import complexity_simulate
from .transition_matrix import transition_matrix, transition_matrix_simulate

# Entropy
from .entropy_shannon import entropy_shannon
from .fractal_correlation import fractal_correlation
from .entropy_approximate import entropy_approximate
from .entropy_fuzzy import entropy_fuzzy
from .entropy_multiscale import entropy_multiscale
from .entropy_sample import entropy_sample

# Fractal
from .fractal_dfa import fractal_dfa
from .fractal_mandelbrot import fractal_mandelbrot


# Aliases
complexity_se = entropy_shannon

complexity_apen = entropy_approximate
complexity_capen = functools.partial(entropy_approximate, corrected=True)

complexity_sampen = entropy_sample
complexity_fuzzyen = entropy_fuzzy

complexity_mse = entropy_multiscale
complexity_fuzzymse = functools.partial(entropy_multiscale, fuzzy=True)
complexity_cmse = functools.partial(entropy_multiscale, composite=True)
complexity_fuzzycmse = functools.partial(entropy_multiscale, composite=True, fuzzy=True)
complexity_rcmse = functools.partial(entropy_multiscale, refined=True)
complexity_fuzzyrcmse = functools.partial(entropy_multiscale, refined=True, fuzzy=True)


complexity_dfa = fractal_dfa
fractal_mfdfa = functools.partial(fractal_dfa, multifractal=True)
complexity_mfdfa = fractal_mfdfa

complexity_d2 = fractal_correlation

complexity_plot = functools.partial(complexity_optimize, show=True)

__all__ = [
    "complexity_embedding",
    "complexity_delay",
    "complexity_dimension",
    "complexity_optimize",
    "complexity_simulate",
    "complexity_r",
    "entropy_shannon",
    "entropy_approximate",
    "entropy_sample",
    "entropy_fuzzy",
    "entropy_multiscale",
    "fractal_dfa",
    "fractal_correlation",
    "fractal_mandelbrot",
    "complexity_se",
    "complexity_apen",
    "complexity_capen",
    "complexity_sampen",
    "complexity_fuzzyen",
    "complexity_mse",
    "complexity_fuzzymse",
    "complexity_cmse",
    "complexity_fuzzycmse",
    "complexity_rcmse",
    "complexity_fuzzyrcmse",
    "complexity_dfa",
    "fractal_mfdfa",
    "complexity_mfdfa",
    "complexity_d2",
    "complexity_plot",
    "transition_matrix",
    "transition_matrix_simulate"
]
