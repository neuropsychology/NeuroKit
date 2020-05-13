"""Submodule for NeuroKit."""
import functools

# Utils
from .complexity_embedding import complexity_embedding
from .complexity_delay import complexity_delay
from .complexity_dimension import complexity_dimension
from .complexity_optimize import complexity_optimize
from .complexity_simulate import complexity_simulate
from .complexity_r import complexity_r

# Entropy
from .entropy_shannon import entropy_shannon
from .entropy_approximate import entropy_approximate
from .entropy_sample import entropy_sample
from .entropy_fuzzy import entropy_fuzzy
from .entropy_multiscale import entropy_multiscale

# Fractal
from .fractal_dfa import fractal_dfa
from .fractal_correlation import fractal_correlation
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
