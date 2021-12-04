"""Submodule for NeuroKit."""
import functools

from .complexity import complexity
from .complexity_attractor import complexity_attractor
from .complexity_embedding import complexity_embedding
from .complexity_hjorth import complexity_hjorth
from .complexity_hurst import complexity_hurst
from .complexity_lempelziv import complexity_lempelziv
from .complexity_lyapunov import complexity_lyapunov
from .complexity_recurrence import complexity_recurrence
from .complexity_rqa import complexity_rqa
from .complexity_rr import complexity_rr
from .complexity_simulate import complexity_simulate
from .entropy_approximate import entropy_approximate
from .entropy_coalition import entropy_coalition
from .entropy_cumulative_residual import entropy_cumulative_residual
from .entropy_differential import entropy_differential
from .entropy_fuzzy import entropy_fuzzy
from .entropy_multiscale import entropy_multiscale
from .entropy_permutation import entropy_permutation
from .entropy_range import entropy_range
from .entropy_sample import entropy_sample
from .entropy_shannon import entropy_shannon
from .entropy_spectral import entropy_spectral
from .entropy_svd import entropy_svd
from .fractal_correlation import fractal_correlation
from .fractal_dfa import fractal_dfa
from .fractal_higuchi import fractal_higuchi
from .fractal_katz import fractal_katz
from .fractal_mandelbrot import fractal_mandelbrot
from .fractal_nld import fractal_nld
from .fractal_petrosian import fractal_petrosian
from .fractal_psdslope import fractal_psdslope
from .fractal_sda import fractal_sda
from .fractal_sevcik import fractal_sevcik
from .information_fisher import fisher_information
from .information_mutual import mutual_information
from .optim_complexity_delay import complexity_delay
from .optim_complexity_dimension import complexity_dimension
from .optim_complexity_k import complexity_k
from .optim_complexity_optimize import complexity_optimize
from .optim_complexity_tolerance import complexity_tolerance
from .TODO_entropy_wiener import entropy_wiener
from .transition_matrix import transition_matrix, transition_matrix_simulate

# Aliases
complexity_se = entropy_shannon
complexity_diffen = entropy_differential
complexity_cren = entropy_cumulative_residual

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

complexity_pe = entropy_permutation
complexity_wpe = functools.partial(entropy_permutation, weighted=True)
complexity_mspe = functools.partial(entropy_permutation, scale="default")

complexity_dfa = fractal_dfa
fractal_mfdfa = functools.partial(fractal_dfa, multifractal=True)
complexity_mfdfa = fractal_mfdfa

complexity_lzc = complexity_lempelziv
complexity_plzc = functools.partial(complexity_lzc, permutation=True)
complexity_mplzc = functools.partial(complexity_lzc, multiscale=True)

complexity_cd = fractal_correlation

complexity_plot = functools.partial(complexity_optimize, show=True)

__all__ = [
    "complexity",
    "complexity_attractor",
    "complexity_embedding",
    "complexity_recurrence",
    "complexity_delay",
    "complexity_dimension",
    "complexity_optimize",
    "complexity_simulate",
    "complexity_hjorth",
    "complexity_hurst",
    "complexity_tolerance",
    "complexity_lempelziv",
    "complexity_lzc",
    "complexity_plzc",
    "complexity_mplzc",
    "complexity_lyapunov",
    "complexity_mfdfa",
    "complexity_cd",
    "complexity_plot",
    "complexity_se",
    "complexity_apen",
    "complexity_capen",
    "complexity_cren",
    "complexity_diffen",
    "complexity_k",
    "complexity_sampen",
    "complexity_fuzzyen",
    "complexity_mse",
    "complexity_fuzzymse",
    "complexity_cmse",
    "complexity_fuzzycmse",
    "complexity_rcmse",
    "complexity_fuzzyrcmse",
    "complexity_pe",
    "complexity_wpe",
    "complexity_mspe",
    "complexity_dfa",
    "complexity_rr",
    "complexity_rqa",
    "entropy_shannon",
    "entropy_differential",
    "entropy_cumulative_residual",
    "entropy_approximate",
    "entropy_coalition",
    "entropy_sample",
    "entropy_spectral",
    "entropy_svd",
    "entropy_fuzzy",
    "entropy_multiscale",
    "entropy_wiener",
    "entropy_permutation",
    "entropy_range",
    "fisher_information",
    "fractal_dfa",
    "fractal_correlation",
    "fractal_higuchi",
    "fractal_katz",
    "fractal_petrosian",
    "fractal_sevcik",
    "fractal_mandelbrot",
    "fractal_mfdfa",
    "fractal_nld",
    "fractal_psdslope",
    "fractal_sda",
    "mutual_information",
    "transition_matrix",
    "transition_matrix_simulate",
]
