"""
Totani Helpers: Core utilities for Fermi-LAT analysis validation.

This package provides the core functionality for the Totani (2025) validation
analysis, including data I/O, template fitting, MCMC analysis, and plotting.

Modules
-------
totani_io : Data I/O for Fermi-LAT products and templates
fit_utils : Template fitting and likelihood functions
mcmc_io : MCMC result handling and coefficient extraction
mcmc_plotter : Spectral analysis utilities for Figures 2-3
cellwise_fit : Spatial cell-by-cell fitting
bubbles_templates : Fermi bubble geometry and masking
plotting : Plotting utilities with Totani (2025) styling

Example
-------
>>> from totani_helpers import totani_io, fit_utils
>>> counts, hdr, Emin, Emax, Ectr, dE = totani_io.read_counts_and_ebounds('counts.fits')
>>> templates = totani_io.load_all_templates(template_dir='templates/')
"""

__version__ = "0.1.0"

from .plotting import plot_E2_dnde_multi_totani, totani_component_style
from .bubbles_templates import build_flat_counts_template, iterate_bubbles_masks

__all__ = [
    "plot_E2_dnde_multi_totani",
    "totani_component_style",
    "build_flat_counts_template",
    "iterate_bubbles_masks",
]
