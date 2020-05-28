"""Submodule for NeuroKit."""

from .standardize import standardize
from .hdi import hdi
from .mad import mad
from .correlation import cor
from .density import density
from .distance import distance
from .rescale import rescale
from .fit_loess import fit_loess
from .fit_polynomial import fit_polynomial
from .fit_polynomial import fit_polynomial_findorder
from .fit_mixture import fit_mixture
from .fit_error import fit_error
from .fit_error import fit_mse
from .fit_error import fit_rmse
from .fit_error import fit_r2
from .mutual_information import mutual_information
from .summary import summary_plot

__all__=["standardize", "hdi", "mad", "cor", "density", "distance", "rescale", "fit_loess", "fit_polynomial", 
         "fit_polynomial_findorder", "fit_mixture", "fit_error", "fit_mse", "fit_rmse", "fit_r2", "mutual_information", "summary_plot"]