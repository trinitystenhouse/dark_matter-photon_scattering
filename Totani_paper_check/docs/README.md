# Fermi-LAT Analysis: Totani (2025) Validation

Independent reimplementation and validation of the Fermi-LAT gamma-ray analysis presented in **Totani (2025)**, focusing on the Galactic center excess and Fermi bubble morphology.

## Overview

This package provides tools for:
- Processing Fermi-LAT photon data with custom energy binning and spatial selections
- Building multi-component gamma-ray emission templates (point sources, diffuse backgrounds, dark matter candidates)
- Performing Bayesian MCMC fits to extract component normalizations
- Reproducing key figures from Totani (2025) for validation

The analysis uses a template-fitting approach with physically motivated background models and explores potential dark matter annihilation signals in the Galactic center region.

## Requirements

- **Python**: ≥3.10
- **Conda/Miniconda**: For environment management
- **Fermitools**: NASA's Fermi Science Tools (installed via conda)
- **Dependencies**: See `requirements_fermi.txt` and `requirements_analysis.txt` in repo root

## Installation

From the repository root:

```bash
source Totani_paper_check/setup.sh
```

This script:
1. Creates/activates the `fermi` conda environment
2. Installs Fermi Science Tools and analysis dependencies
3. Installs this package (`totani-paper-check`) in editable mode
4. Exports `REPO_PATH` for data access

## Data Structure

The analysis expects Fermi-LAT data products under `${REPO_PATH}/fermi_data/totani/`:

```
fermi_data/totani/
├── raw/                          # Downloaded photon/spacecraft files
├── processed/
│   ├── counts_ccube_*.fits       # Counts cubes (energy binned)
│   ├── expcube_*.fits            # Exposure cubes
│   ├── ltcube_*.fits             # Livetime cubes
│   └── templates/                # Emission templates
│       ├── ps_template.fits
│       ├── gas_template.fits
│       ├── ics_template.fits
│       ├── iso_template.fits
│       ├── nfw_template.fits
│       ├── loopI_template.fits
│       └── bubbles_*.fits
```

## Workflow

### 1. Data Download

Download Fermi-LAT photon and spacecraft data:

```bash
cd Totani_paper_check/data_download
bash download_fermi_totani.sh
```

This retrieves data from the Fermi Science Support Center matching the Totani (2025) selection criteria.

### 2. Data Processing

Process raw photon files into analysis-ready data products:

```bash
bash data_download/process_totani.sh
```

Generates counts cubes, exposure maps, and livetime cubes using Fermi Science Tools.

### 3. Sanity Checks

Validate data products before template generation:

```bash
python sanity_checks/sanity_checks.py
```

Produces diagnostic plots in `sanity_checks/sanity_plots/`.

### 4. Template Generation

Build emission templates for multi-component fitting. **Must run in order:**

```bash
cd make_templates

# 1. Mask extended sources (required first)
python mask_extended_sources.py

# 2. Point source template
python build_ps_template.py

# 3. Diffuse backgrounds (gas, inverse Compton, isotropic)
python build_gas_template.py
python build_ics_template.py
python build_iso_template.py

# 4. Dark matter candidate (NFW profile)
python build_nfw_template.py --rho-power 2.5 --norm pole

# 5. Loop I (local superbubble)
python build_loopI_template.py

# 6. Fermi bubbles
python build_bubbles_templates.py
```

Templates are saved to `${REPO_PATH}/fermi_data/totani/processed/templates/`.

### 5. MCMC Fitting

Perform Bayesian MCMC fits to extract component normalizations:

```bash
cd mcmc

# Single energy bin
python run_mcmc.py --energy-bin 2 --nwalkers 32 --nsteps 5000 --burn 1000

# Batch processing across energy bins
python run_mcmc_from_config.py configs/base.json
```

Results are saved as compressed `.npz` files containing posterior samples and best-fit coefficients.

### 6. Figure Reproduction

Reproduce figures from Totani (2025):

```bash
cd figures

# Figure 1: Sky maps
python make_totani_fig1_maps.py

# Figures 2-3: Energy spectra
python make_figures_from_config.py configs/fig2_3.json

# Figure 8: Component contributions
python make_totani_fig8.py

# Figure 9: Likelihood scan
python make_totani_fig9_likelihood.py

# Figure 11: Cellwise analysis
python make_fig11.py

# Figure 12: Posterior distributions
python make_fig12.py
```

## Template Validation

The `check_templates/` directory contains diagnostic scripts for validating template construction:

```bash
cd check_templates

# Check individual components
python check_nfw.py
python check_loopI.py
python check_bubbles_posneg.py

# Run full systematics suite
python run_systematics_suite.py
```

## Key Modules

### `totani_helpers`

Core utilities for the analysis:

- **`totani_io.py`**: Load Fermi data products and templates
- **`fit_utils.py`**: Likelihood functions and fitting utilities
- **`mcmc_io.py`**: MCMC output handling and coefficient extraction
- **`mcmc_plotter.py`**: MCMC-specific routines for spectral fitting
- **`cellwise_fit.py`**: Spatial cell-by-cell fitting
- **`bubbles_templates.py`**: Fermi bubble geometry and masking
- **`plotting.py`**: Plotting utilities with Totani (2025) styling

## Implementation Notes

### Template Normalization

Physical templates (gas, ICS, point sources, isotropic) are normalized to their baseline flux predictions. Spatial templates (NFW, Loop I, Fermi bubbles) are shape-only, with fitted coefficients representing physical flux in units of [ph cm⁻² s⁻¹ sr⁻¹ MeV⁻¹].

### Validation Status

This is an independent implementation intended to validate the methodology of Totani (2025). The code includes configurable parameters for:
- Energy binning scheme and bin edges
- Extended source masking criteria  
- NFW profile parameters (slope, normalization point)
- MCMC sampler settings (walkers, burn-in, convergence criteria)

These parameters can be adjusted via command-line arguments to match the published analysis. Comparison with Totani (2025) results is ongoing.

## Citation

If you use this code, please cite:

- **Totani (2025)**: [Full citation to be added]
- **Fermi-LAT Collaboration**: [Relevant Fermi-LAT papers]

## Contact

For questions about this implementation, please contact [your contact information].

## License

[Specify license if applicable]
