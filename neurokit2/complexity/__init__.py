"""Submodule for NeuroKit."""

import functools

from .complexity import complexity
from .complexity_decorrelation import complexity_decorrelation
from .complexity_hjorth import complexity_hjorth
from .complexity_irreversibility import complexity_irreversibility
from .complexity_jensenshannon import complexity_jensenshannon
from .complexity_lempelziv import complexity_lempelziv
from .complexity_lyapunov import complexity_lyapunov
from .complexity_relativeroughness import complexity_relativeroughness
from .complexity_rqa import complexity_rqa
from .entropy_angular import entropy_angular
from .entropy_approximate import entropy_approximate
from .entropy_attention import entropy_attention
from .entropy_bubble import entropy_bubble
from .entropy_coalition import entropy_coalition
from .entropy_correctedconditional import entropy_correctedconditional
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
from .entropy_wiener import entropy_wiener
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
complexity_mspe = functools.partial(entropy_multiscale, method="MSPEn")
complexity_cmse = functools.partial(entropy_multiscale, method="CMSEn")
complexity_rcmse = functools.partial(entropy_multiscale, method="RCMSEn")
complexity_fuzzymse = functools.partial(entropy_multiscale, fuzzy=True)
complexity_fuzzycmse = functools.partial(entropy_multiscale, method="CMSEn", fuzzy=True)
complexity_fuzzyrcmse = functools.partial(entropy_multiscale, method="RCMSEn", fuzzy=True)


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
    "complexity_irreversibility",
    "complexity_jensenshannon",
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
    "entropy_correctedconditional",
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


# =============================================================================
# Roadmap: potential additions
# =============================================================================
# From a survey of the literature (2021-2026) and of Julia, R and Python packages (Nov 2026).
# Rough effort: S(mall), M(edium), L(arge). Packages are listed as references to validate
# against (pyhctsa is GPL-3.0 and pyrpde EUPL: implement from the papers rather than porting).
#
# Extensions of existing functions
# - entropy_permutation(): tie handling, which matters for quantized signals (e.g., RR intervals
#   in ms): modified PE, with ties as patterns of their own (Bian et al., 2012, PRE 85:021906),
#   and amplitude-aware PE (Azami & Escudero, 2016, Comput Methods Programs Biomed 128:40).
#   Reference: EntropyHub PermEn(Typex=...). S.
# - complexity_jensenshannon(): Tsallis and Renyi complexity-entropy curves (Ribeiro et al., 2017,
#   PRE 95:062106; Jauregui et al., 2018, Physica A 498:74), and the ordinal-pattern Fisher
#   information of the Fisher-Shannon causality plane (Olivares et al., 2012, Physica A 391:2518),
#   whose value depends on how the patterns are ordered. Reference: ordpy, statcomp. S.
# - complexity_irreversibility(): Costa's multiscale time asymmetry index (Costa et al., 2005,
#   PRL 95:198102). S.
#
# Valuable, moderate effort
# - Visibility graphs, natural (Lacasa et al., 2008, PNAS 105:4972) and horizontal (Luque et al.,
#   2009, PRE 80:046103): exponent of the degree distribution (ln(3/2) for i.i.d. noise) and degree
#   entropy. Parameter-free, used in EEG and HRV. Reference: ts2vg (MIT). M.
# - 0-1 test for chaos (Gottwald & Melbourne, 2004, Proc R Soc A 460:603; 2009, SIAM J Appl Dyn
#   Syst 8:129). The plain test also gives K close to 1 for noise: pair it with the surrogate-based
#   Chaos Decision Tree (Toker et al., 2020, Commun Biol 3:11). Reference: R Chaos01. S-M.
# - Wavelet entropy (Rosso et al., 2001, J Neurosci Methods 105:65): Shannon entropy of the
#   relative energies of the wavelet decomposition levels (PyWavelets). S.
# - Recurrence network measures (Donner et al., 2010, New J Phys 12:033025): transitivity,
#   average path length and clustering of recurrence_matrix(). Reference: pyunicorn. M.
# - Nonlinear prediction / empirical dynamic modelling (Sugihara & May, 1990, Nature 344:734):
#   simplex forecast skill as a function of the horizon, and S-map nonlinearity (theta).
#   Reference: pyEDM, rEDM. M.
# - Recurrence period density entropy (RPDE; Little et al., 2007, BioMed Eng OnLine 6:23), a
#   standard feature in voice analysis, distinct from RQA. S-M.
# - fractal_correlation(): Takens' maximum-likelihood estimator of D2 (with confidence intervals)
#   and generalized dimensions D_q. Reference: FractalDimensions.jl, nonlinearTseries. S-M.
#
# To watch (niche or recent)
# - Rescaled auto-density (RAD), a distance-to-criticality index (Harris, Gollo & Fulcher, 2024,
#   PRX 14:031021). S.
# - Recurrence microstates entropy (Corso et al., 2018, Chaos 28:083108); the tolerance that
#   maximizes it could also be a default for complexity_rqa(). Reference:
#   RecurrenceMicrostatesAnalysis.jl. M.
# - Persistent entropy (topological data analysis), e.g., from the 0-dimensional sublevel-set
#   persistence of the signal (Rucco et al., 2016). Reference: giotto-tda, teaspoon. M.
# - Wavelet-leader multifractal analysis, log-cumulants c1 and c2 (Wendt, Abry & Jaffard, 2007,
#   IEEE Signal Process Mag 24:38), an alternative to MFDFA. Reference: pymultifracs. L.
# - Chhabra & Jensen (1989) direct multifractal spectrum, as an option of fractal_tmf() to match
#   Bell et al. (2019) (see the note in its docstring). Needs a positive measure from the signal
#   (e.g., absolute increments) and keeps only the q whose fits have r >= 0.995. M.
# - Whittle maximum-likelihood estimator of the Hurst exponent of fractional Gaussian noise. S-M.
# - Bias-corrected Shannon entropy estimators (Miller-Madow, Chao-Shen, shrinkage), as an option
#   of the symbolic entropies on short signals. Reference: ComplexityMeasures.jl. S.
# - Tone-entropy of heart rate (Oida et al., 1997, J Appl Physiol 82:1794). S.
# - Functional excitation-inhibition ratio (fE/I; Bruining et al., 2020, Sci Rep,
#   doi:10.1038/s41598-020-65500-4), rather for the EEG module. M.
