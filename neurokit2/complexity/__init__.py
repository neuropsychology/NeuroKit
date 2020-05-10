"""Submodule for NeuroKit."""
import functools

from .embedding import embedding
from .embedding_delay import embedding_delay
from .embedding_dimension import embedding_dimension
from .embedding_concurrent import embedding_concurrent

from .entropy_shannon import entropy_shannon
from .entropy_approximate import entropy_approximate
from .entropy_sample import entropy_sample
from .entropy_fuzzy import entropy_fuzzy
from .entropy_multiscale import entropy_multiscale

from .fractal_dfa import fractal_dfa
from .fractal_correlation import fractal_correlation
from .fractal_mandelbrot import fractal_mandelbrot

from .complexity_simulate import complexity_simulate


# Aliases
complexity_se = entropy_shannon
complexity_apen = entropy_approximate
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
