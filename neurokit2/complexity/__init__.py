"""Submodule for NeuroKit."""
import functools

from .complexity import complexity
from .complexity_decorrelation import complexity_decorrelation
from .complexity_hjorth import complexity_hjorth
from .complexity_lempelziv import complexity_lempelziv
from .complexity_lyapunov import complexity_lyapunov
from .complexity_relativeroughness import complexity_relativeroughness
from .complexity_rqa import complexity_rqa
from .entropy_angular import entropy_angular
from .entropy_approximate import entropy_approximate
from .entropy_attention import entropy_attention
from .entropy_bubble import entropy_bubble
from .entropy_coalition import entropy_coalition
from .entropy_cosinesimilarity import entropy_cosinesimilarity
from .entropy_cumulativeresidual import entropy_cumulativeresidual
from .entropy_differential import entropy_differential
from .entropy_dispersion import entropy_dispersion
from .entropy_distribution import entropy_distribution
from .entropy_fuzzy import entropy_fuzzy
from .entropy_grid import entropy_grid
from .entropy_hierarchical import entropy_hierarchical
from .entropy_increment import entropy_increment
from .entropy_kl import entropy_kl
from .entropy_kolmogorov import entropy_kolmogorov
from .entropy_maximum import entropy_maximum
from .entropy_multiscale import entropy_multiscale
from .entropy_ofentropy import entropy_ofentropy
from .entropy_permutation import entropy_permutation
from .entropy_phase import entropy_phase
from .entropy_power import entropy_power
from .entropy_quadratic import entropy_quadratic
from .entropy_range import entropy_range
from .entropy_rate import entropy_rate
from .entropy_renyi import entropy_renyi
from .entropy_sample import entropy_sample
from .entropy_shannon import entropy_shannon
from .entropy_shannon_joint import entropy_shannon_joint
from .entropy_slope import entropy_slope
from .entropy_spectral import entropy_spectral
from .entropy_svd import entropy_svd
from .entropy_symbolicdynamic import entropy_symbolicdynamic
from .entropy_tsallis import entropy_tsallis
from .fractal_correlation import fractal_correlation
from .fractal_density import fractal_density
from .fractal_dfa import fractal_dfa
from .fractal_higuchi import fractal_higuchi
from .fractal_hurst import fractal_hurst
from .fractal_katz import fractal_katz
from .fractal_linelength import fractal_linelength
from .fractal_nld import fractal_nld
from .fractal_petrosian import fractal_petrosian
from .fractal_psdslope import fractal_psdslope
from .fractal_sda import fractal_sda
from .fractal_sevcik import fractal_sevcik
from .fractal_tmf import fractal_tmf
from .information_fisher import fisher_information
from .information_fishershannon import fishershannon_information
from .information_gain import information_gain
from .information_mutual import mutual_information
from .optim_complexity_delay import complexity_delay
from .optim_complexity_dimension import complexity_dimension
from .optim_complexity_k import complexity_k
from .optim_complexity_optimize import complexity_optimize
from .optim_complexity_tolerance import complexity_tolerance
from .TODO_entropy_wiener import entropy_wiener
from .utils_complexity_attractor import complexity_attractor
from .utils_complexity_coarsegraining import complexity_coarsegraining
from .utils_complexity_embedding import complexity_embedding
from .utils_complexity_ordinalpatterns import complexity_ordinalpatterns
from .utils_complexity_simulate import complexity_simulate
from .utils_complexity_symbolize import complexity_symbolize
from .utils_fractal_mandelbrot import fractal_mandelbrot
from .utils_recurrence_matrix import recurrence_matrix

# Aliases
complexity_se = entropy_shannon
complexity_diffen = entropy_differential
complexity_cren = entropy_cumulativeresidual

complexity_apen = entropy_approximate
complexity_capen = functools.partial(entropy_approximate, corrected=True)

complexity_atten = entropy_attention

complexity_sampen = entropy_sample
complexity_fuzzyen = entropy_fuzzy
complexity_fuzzyapen = functools.partial(entropy_fuzzy, approximate=True)

complexity_pe = entropy_permutation
complexity_wpe = functools.partial(entropy_permutation, weighted=True)

complexity_mse = entropy_multiscale
complexity_mspe = functools.partial(entropy_multiscale, scale="MSPEn")
complexity_cmse = functools.partial(entropy_multiscale, method="CMSEn")
complexity_rcmse = functools.partial(entropy_multiscale, method="RCMSEn")
complexity_fuzzymse = functools.partial(entropy_multiscale, fuzzy=True)
complexity_fuzzycmse = functools.partial(entropy_multiscale, method="CMSEn", fuzzy=True)
complexity_fuzzyrcmse = functools.partial(
    entropy_multiscale, method="RCMSEn", fuzzy=True
)


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
    "complexity_coarsegraining",
    "complexity_ordinalpatterns",
    "complexity_symbolize",
    "complexity_decorrelation",
    "recurrence_matrix",
    "complexity_delay",
    "complexity_dimension",
    "complexity_optimize",
    "complexity_simulate",
    "complexity_hjorth",
    "fractal_hurst",
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
    "complexity_atten",
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
    "complexity_relativeroughness",
    "complexity_rqa",
    "entropy_angular",
    "entropy_maximum",
    "entropy_shannon",
    "entropy_shannon_joint",
    "entropy_power",
    "entropy_rate",
    "entropy_tsallis",
    "entropy_renyi",
    "entropy_kolmogorov",
    "entropy_attention",
    "entropy_ofentropy",
    "entropy_slope",
    "entropy_increment",
    "entropy_differential",
    "entropy_kl",
    "entropy_distribution",
    "entropy_symbolicdynamic",
    "entropy_cumulativeresidual",
    "entropy_approximate",
    "entropy_quadratic",
    "entropy_bubble",
    "entropy_coalition",
    "entropy_sample",
    "entropy_phase",
    "entropy_dispersion",
    "entropy_grid",
    "entropy_spectral",
    "entropy_svd",
    "entropy_fuzzy",
    "complexity_fuzzyapen",
    "entropy_multiscale",
    "entropy_hierarchical",
    "entropy_wiener",
    "entropy_permutation",
    "entropy_range",
    "entropy_cosinesimilarity",
    "fisher_information",
    "fishershannon_information",
    "fractal_dfa",
    "fractal_correlation",
    "fractal_density",
    "fractal_higuchi",
    "fractal_katz",
    "fractal_linelength",
    "fractal_petrosian",
    "fractal_sevcik",
    "fractal_mandelbrot",
    "fractal_mfdfa",
    "fractal_tmf",
    "fractal_nld",
    "fractal_psdslope",
    "fractal_sda",
    "mutual_information",
    "information_gain",
]
